# The container ONE SN22 submission runs in, inside the sealed room.
#
# SN60 needs no equivalent: its inner containers are third-party `ghcr.io/bitsec-ai/<project>`
# images from the Bitsec upstream. An SN22 submission is a plain `agent.py`, so the container it
# runs in has to be built here.
#
# It carries exactly two things beyond Python:
#
#   1. `sn22_relay.py` — the SAME client module the local sandbox copies into a run directory. The
#      room mounts the bundle READ-ONLY, so nothing can be added to it at run time; without this
#      image shipping the module, every submission would die on `import sn22_relay` in its first
#      real round, and it would read as a bad agent rather than a missing file.
#   2. Nothing else. No package index, no build tools, no shell utilities a submission could reach
#      for. A submission runs on the standard library, which is what it is told and what it gets.
#
# The room runs this with `--network <sealed>`, `--read-only`, `--cap-drop ALL`,
# `--security-opt no-new-privileges` and a memory/CPU ceiling. This image does not need to defend
# itself; it needs to contain nothing worth reaching.
ARG PYTHON_BASE=python@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf
FROM ${PYTHON_BASE}

# On PYTHONPATH rather than pip-installed: the module is one reviewed file, and `pip install` into
# an image that runs untrusted code adds a package manager the agent could otherwise invoke.
COPY sn22_relay.py /opt/kata/sn22_relay.py
ENV PYTHONPATH=/opt/kata \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Never root. The room drops capabilities anyway; this is the layer that means there are none to
# drop in the first place.
RUN useradd --create-home --uid 10001 kata
USER 10001

# No CMD. The room supplies `python /bundle/agent.py` explicitly, so this image cannot run anything
# on its own if it is ever pulled by mistake.
