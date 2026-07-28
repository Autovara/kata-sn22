# The container ONE SN22 submission runs in, inside the sealed room.
#
# SN60 needs no equivalent: its inner containers are third-party `ghcr.io/bitsec-ai/<project>`
# images from the Bitsec upstream. An SN22 submission is a plain `agent.py`, so the container it
# runs in has to be built here.
#
# It carries exactly three things beyond Python:
#
#   1. `kata_sn22_sdk` — the reviewed SDK a submission imports: the version-2 task and answer
#      models, the typed broker client, and the `Agent` base class. There is no credential on it
#      and no way to ask for one.
#   2. The harness, which is this image's entry point. It imports `/bundle/agent.py`, runs one
#      task, and frames one answer in the exact fields the pinned scorer reads. Framing is the
#      lane's job, not the submission's: a per-agent serialiser would make "both contestants were
#      asked the same thing" depend on two strangers' JSON.
#   3. `sn22_relay.py` — the version-1 client the local sandbox serves. Kept so a submission
#      calibrated against the older interface does not die on `import sn22_relay`.
#
# ...and then it carries LESS than the base image did. The final stage deletes pip, ensurepip,
# setuptools, apt, dpkg, curl and wget. That is the point of this image: not to defend itself, but
# to contain nothing worth reaching. An untrusted agent with a package manager is one network path
# away from running code that was never reviewed.
#
# The room runs it with `--network <sealed-internal>`, `--read-only`, `--cap-drop ALL`,
# `--security-opt no-new-privileges`, a non-root UID and memory/CPU/PID ceilings.

ARG PYTHON_BASE=python@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
FROM ${PYTHON_BASE}

# On PYTHONPATH rather than pip-installed. `pip install` into an image that runs untrusted code
# leaves the package manager behind, and the whole point below is that it is not there.
COPY kata_sn22_sdk /opt/kata/kata_sn22_sdk
COPY sn22_relay.py /opt/kata/sn22_relay.py

ENV PYTHONPATH=/opt/kata \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Strip the toolchain. Done as root, in one layer, before the image ever drops to a normal user --
# a later layer cannot restore what an earlier one deleted, and the image digest covers the result.
#
# `find -delete` rather than a fixed path list: the site-packages directory carries the minor
# version, and a hard-coded `python3.13` would silently stop deleting anything on the next base
# bump while still looking correct.
RUN set -eux; \
    find /usr/local/lib -maxdepth 3 -type d \
        \( -name 'pip' -o -name 'pip-*' -o -name 'setuptools' -o -name 'setuptools-*' \
           -o -name 'pkg_resources' -o -name 'ensurepip' -o -name '_distutils_hack' \) \
        -exec rm -rf {} +; \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.* \
          /usr/local/bin/easy_install* /usr/local/bin/wheel; \
    rm -f /usr/bin/apt /usr/bin/apt-get /usr/bin/apt-cache /usr/bin/apt-key /usr/bin/apt-config \
          /usr/bin/dpkg /usr/bin/dpkg-deb /usr/bin/dpkg-query /usr/bin/dpkg-split \
          /usr/bin/curl /usr/bin/wget /usr/bin/nc /usr/bin/netcat /usr/bin/telnet \
          /usr/bin/ssh /usr/bin/scp /usr/bin/sftp /usr/bin/rsync /usr/bin/git; \
    rm -rf /var/lib/apt/lists/* /var/cache/apt /root/.cache; \
    useradd --create-home --uid 10001 kata; \
    # Fail the BUILD, not a duel, if any of it survived.
    ! command -v pip; ! command -v pip3; ! command -v apt-get; ! command -v curl; \
    ! python -c 'import pip'; \
    ! python -c 'import ensurepip'; \
    python -c 'import kata_sn22_sdk, kata_sn22_sdk.harness, sn22_relay'

# Never root. The room drops capabilities anyway; this is the layer that means there are none to
# drop in the first place.
USER 10001

# The harness, not the submission. The room appends the bundle path. A submission is imported by
# this entry point rather than executed as a program, so every answer is framed by one reviewed
# function.
ENTRYPOINT ["python", "-m", "kata_sn22_sdk.harness"]
CMD ["/bundle/agent.py"]
