"""The candidate execution sandbox (plan §6.1–§6.2, SN22-4).

A submission is code from a stranger, run on the machine that holds every credential in the system.
Everything here follows from taking that literally.

Two layers, and both matter because either alone is insufficient:

* **The environment is constructed, not filtered.** ``candidate_env`` builds a fresh dict from
  nothing. A filtered copy of ``os.environ`` leaks whatever the filter forgot, and what it forgets
  is by definition the variable added last — the new provider key. There is no denylist here to
  fall out of date.
* **The filesystem and network are removed, not merely forbidden.** Under ``bwrap`` the credential
  files, the operator's home and ``/srv`` are ABSENT from the mount namespace rather than present
  and unreadable, and there is no network namespace at all. A path-traversal bug in the agent then
  has nothing to traverse to.

Search still works inside that, because the relay is a **unix socket** in the run directory rather
than an HTTP endpoint (:mod:`kata_sn22.relay_server`). A unix socket is a filesystem object, so it
survives ``--unshare-net`` — which is what lets the sandbox keep no network at all while the agent
can still search. The agent never opens it: the lane copies ``sn22_relay.py`` into the run directory
and the agent imports it, so no submission has any reason to touch ``socket`` and the screen's ban
on it can stay absolute.

``build_argv`` is a pure function so the policy can be asserted without a host that can sandbox;
``run_candidate`` needs a real one and REFUSES rather than falling back. A submission that could not
be isolated has not been evaluated, and scoring it anyway would be the whole point thrown away.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

BWRAP = os.environ.get("KATA_SN22_BWRAP", "/usr/bin/bwrap")
SUDO = os.environ.get("KATA_SN22_SUDO", "/usr/bin/sudo")

#: nobody/nogroup: the candidate owns nothing on the host.
SANDBOX_UID = 65534
SANDBOX_GID = 65534

#: The minimum read-only system needed to start an interpreter. ``/srv``, ``/home``, ``/root`` and
#: ``/etc`` are deliberately absent — that is what makes the §6.2 credential probes fail.
_BASE_RO_PATHS = ("/usr", "/bin", "/sbin", "/lib", "/lib64")

#: Paths a candidate must never reach. Used by the boundary tests as the things to try to read.
FORBIDDEN_PATHS = (
    "/srv/kata-bot/.env",
    "/srv/kata-subnets/registry.json",
    "/srv/kata-subnets/approvals",
    "/root/.ssh/id_rsa",
    "/home/ubuntu/.ssh/id_rsa",
    "/etc/shadow",
)


class SandboxUnavailable(Exception):
    """This host cannot isolate a candidate. NOT a pass: never run the submission unconfined."""


@dataclass(frozen=True)
class SandboxLimits:
    max_wall_seconds: int = 120
    max_memory_bytes: int = 1024 * 1024 * 1024
    max_processes: int = 128
    max_file_bytes: int = 64 * 1024 * 1024
    max_output_bytes: int = 256 * 1024


def candidate_env(*, task_input: dict, relay_endpoint: str, capability: str,
                  workdir: str) -> dict[str, str]:
    """The COMPLETE environment a candidate sees. Constructed from nothing, never filtered.

    The task itself also arrives on stdin; the few variables here exist so an agent can find its
    relay without parsing. Note what is absent: every provider key, the GitHub token, the webhook
    secret, and anything else the host happens to be holding.

    ``PYTHONPATH`` is the one addition, and it points at exactly one directory: the run directory,
    where the lane placed ``sn22_relay.py``. Without it the agent could not import the relay client
    at all — ``python agent.py`` puts the *script's* directory on ``sys.path``, not the working
    directory — and an agent that cannot reach the relay would have to open a socket itself, which
    is the thing the static screen exists to refuse. It is a single, lane-owned path, not an
    inherited one; nothing of the lane's own code is reachable through it.
    """
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LC_ALL": "C.UTF-8",
        "PYTHONPATH": workdir,
        "PYTHONDONTWRITEBYTECODE": "1",   # no __pycache__ in a directory the next run inherits
        "SN22_PROTOCOL_VERSION": str(task_input.get("protocol_version", 1)),
        "SN22_TASK_ID": str(task_input.get("task_id", "")),
        "SN22_RELAY_ENDPOINT": relay_endpoint,
        "SN22_RELAY_CAPABILITY": capability,
    }


def available() -> bool:
    return Path(BWRAP).exists()


#: Where a submission's bundle is staged inside the run directory. Read-only once mounted.
BUNDLE_DIRNAME = ".submission"


def stage_bundle(bundle_dir: str | Path, workdir: str | Path) -> Path:
    """Copy a submission bundle into the run directory and return its ``agent.py``.

    Staging rather than binding the submission where it lies, for two independent reasons:

    * **bwrap resolves bind sources after dropping privileges**, so a submission checked out under
      an operator's home is simply unreachable — and the failure surfaces as a non-zero exit, which
      would be scored as the CANDIDATE crashing. Misattributing an infrastructure failure to a
      contestant is exactly what plan §5.4 forbids, and it is invisible when it happens.
    * **The bundle is all the agent should see.** Binding its original directory would expose
      whatever sits beside it in the checkout.

    Symlinks and compiled artefacts are dropped rather than copied: a symlink is how a bundle
    reaches out of itself, and a stale ``.pyc`` is code that no reviewer read.
    """
    source = Path(bundle_dir).expanduser().resolve()
    target = Path(workdir).expanduser().resolve() / BUNDLE_DIRNAME
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            continue
        relative = path.relative_to(source)
        if any(part in ("__pycache__", ".git") for part in relative.parts):
            continue
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            destination.chmod(0o755)
        elif path.is_file() and path.suffix not in (".pyc", ".pyo"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            destination.chmod(0o644)
    target.chmod(0o755)
    return target / "agent.py"


def build_argv(agent_py: str | Path, *, workdir: str | Path, limits: SandboxLimits,
               as_root_launcher: bool = True, env: dict[str, str] | None = None) -> list[str]:
    """The exact command that runs one submission. Pure; no side effects.

    ``workdir`` is the ONLY writable path, and it is bound at the same location inside so any path
    the agent prints stays meaningful outside. A bundle staged inside it is re-bound READ-ONLY on
    top, so an agent cannot rewrite its own entry point between tasks — which would mean task 2 was
    scored on code that never faced task 1.

    When ``env`` is given it is applied INSIDE the sandbox with ``--clearenv``/``--setenv`` rather
    than by handing a dict to ``subprocess``. That matters under the root launcher: ``sudo`` resets
    the environment by default, so a passed-through ``PYTHONPATH`` or relay capability would simply
    never arrive — silently, and looking exactly like a candidate that failed to import. Setting it
    here makes the candidate's whole environment part of the reviewed command line.
    """
    work = Path(workdir).expanduser().resolve()
    agent = Path(agent_py).expanduser().resolve()

    command: list[str] = []
    if as_root_launcher:
        # Root creates the namespace; the candidate inside is uid 65534 with no capabilities. Same
        # shape as the trusted installer: privilege builds the jail, untrusted code runs in it.
        command += [SUDO, "-n"]
    command += [BWRAP]
    for path in _BASE_RO_PATHS:
        if Path(path).exists():
            command += ["--ro-bind", path, path]
    # ORDER MATTERS: bwrap applies mounts in argument order, so /tmp's tmpfs must be established
    # before any caller path under it, or the tmpfs silently masks the workdir.
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    command += ["--bind", str(work), str(work)]
    staged = work / BUNDLE_DIRNAME
    if staged in agent.parents:
        # Inside the writable workdir, re-mounted read-only. ORDER MATTERS: after the rw bind.
        command += ["--ro-bind", str(staged), str(staged)]
    else:
        # An external agent path. Kept for callers that stage elsewhere; note that bwrap resolves
        # this source AFTER dropping privileges, so it must be world-traversable.
        command += ["--ro-bind", str(agent), str(agent)]
    command += [
        "--chdir", str(work),
        "--unshare-user", "--uid", str(SANDBOX_UID), "--gid", str(SANDBOX_GID),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup",
        # No network namespace at all: DNS and HTTP fail because there is nothing to reach, not
        # because a rule said no.
        "--unshare-net",
        "--new-session",        # no controlling terminal to inject into
        "--die-with-parent",
        "--cap-drop", "ALL",
    ]
    if env is not None:
        command += ["--clearenv"]
        for name in sorted(env):
            command += ["--setenv", name, env[name]]
    return [*command, "--", "/usr/bin/python3", str(agent)]


def _apply_rlimits(limits: SandboxLimits):
    def _set() -> None:  # pragma: no cover - runs in the forked child
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (limits.max_memory_bytes, limits.max_memory_bytes))
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_bytes, limits.max_file_bytes))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))   # no core dump of untrusted memory
    return _set


@dataclass(frozen=True)
class SandboxRun:
    returncode: int
    stdout: bytes
    stderr: str
    timed_out: bool = False
    truncated: bool = False


def fresh_workdir(root: str | Path, name: str) -> Path:
    """A disposable, empty writable surface, recreated per run so nothing survives between them."""
    workdir = Path(root).expanduser().resolve() / name
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    workdir.chmod(0o777)   # the sandbox uid must be able to write here
    # bwrap resolves bind sources AFTER dropping privileges, so every ancestor must be searchable by
    # uid 65534. Only o+x is added: traversable, not listable.
    current = workdir.parent
    stop = Path("/tmp").resolve()
    while current != stop and stop in current.parents:
        try:
            mode = current.stat().st_mode & 0o777
            if not mode & 0o001:
                current.chmod(mode | 0o001)
        except OSError:
            break
        current = current.parent
    return workdir


def run_candidate(agent_py: str | Path, task_input: dict, *, workdir: str | Path,
                  relay_endpoint: str, capability: str,
                  limits: SandboxLimits | None = None,
                  as_root_launcher: bool = True) -> SandboxRun:
    """Run one submission under isolation, or refuse. Never falls back to running unconfined."""
    limits = limits or SandboxLimits()
    if not available():
        raise SandboxUnavailable(
            f"{BWRAP} is absent; refusing to run an untrusted submission unconfined")
    work = Path(workdir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    env = candidate_env(task_input=task_input, relay_endpoint=relay_endpoint,
                        capability=capability, workdir=str(work))
    command = build_argv(agent_py, workdir=work, limits=limits,
                         as_root_launcher=as_root_launcher, env=env)
    try:
        completed = subprocess.run(
            # ``env`` is applied INSIDE the sandbox (see build_argv). What is passed here is only
            # what the LAUNCHER needs, and it is minimal on purpose: sudo and bwrap inherit nothing
            # from the lane's own environment, so a provider key cannot ride in on the launcher.
            command, input=json.dumps(task_input).encode("utf-8"), capture_output=True,
            env={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            timeout=limits.max_wall_seconds, check=False,
            preexec_fn=_apply_rlimits(limits),
        )
    except subprocess.TimeoutExpired:
        return SandboxRun(returncode=124, stdout=b"", stderr="timed out", timed_out=True)
    except OSError as exc:
        raise SandboxUnavailable(f"cannot launch the sandbox: {exc}") from exc

    stderr = completed.stderr.decode("utf-8", errors="replace")
    # bwrap reports its OWN setup failures on stderr and exits non-zero BEFORE it ever executes the
    # payload. Left unclassified, every one of them would be indistinguishable from the candidate
    # crashing — and a lane that scores a mount failure as an agent's fault is a lane whose results
    # depend on the host. Anything prefixed `bwrap:` is the sandbox failing, not the submission.
    if completed.returncode != 0 and stderr.lstrip().startswith("bwrap:"):
        raise SandboxUnavailable(
            "the sandbox could not start, so the submission was NOT executed: "
            + stderr.strip().splitlines()[0])
    if completed.returncode != 0 and "setting up uid map" in stderr:
        raise SandboxUnavailable(
            "cannot create a user namespace (kernel.apparmor_restrict_unprivileged_userns=1). "
            "The launcher must be root; refusing to run unconfined.")
    stdout = completed.stdout
    truncated = len(stdout) > limits.max_output_bytes
    return SandboxRun(returncode=completed.returncode,
                      stdout=stdout[:limits.max_output_bytes] if truncated else stdout,
                      stderr=stderr[:4000], truncated=truncated)
