#!/usr/bin/env bash
# Run a command inside the lm2.6 container with the LM build and the studies tree mounted.
#   ./lmrun.sh <gpu> python synapse_run.py --arm point --out out/foo.npz
# Containers must mount /lm, /ftsz AND /raid/racda/campaigns: the study's out/ is a symlink
# into the campaigns tree, so a run that mounts only the first two writes into a dangling link.
set -euo pipefail
GPU="${1:-0}"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LM="${REPO}/Lattice_Microbes_2.6"
HERE="/ftsz/studies/tcr-pmhc"
# The jLM shipped in build_container has an uncompiled Lattice.pyx, so `from jLM.RDME import Sim`
# fails on an import of jLM.Lattice. A private copy with that module cythonized lives in the
# campaign tree and goes FIRST on the path; the LM tree itself is left untouched (it is not
# writable as this uid anyway). Rebuild it with build_jlm.sh if the container changes.
JLM="/raid/racda/campaigns/tcr_pmhc/jlm"
PP="${JLM}:/lm/build_container:/lm/build_container/pylm:/ftsz/studies/common"
CMD="$(printf "%q " "$@")"   # preserve per-argument quoting through bash -c
exec docker run --rm --gpus "\"device=${GPU}\"" \
    -v "${LM}":/lm -v "${REPO}":/ftsz -v /raid/racda/campaigns:/raid/racda/campaigns -v /raid/racda/opt:/raid/racda/opt:ro -v /raid/racda/tmp:/raid/racda/tmp \
    -w "${HERE}" -u "$(id -u):$(id -g)" \
    -e USER="$(id -un)" -e HOME=/tmp \
    lm2.6:latest \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && \
             export PATH=/raid/racda/opt/texlive/2026/bin/x86_64-linux:\$PATH && export TMPDIR=/raid/racda/tmp && export PYTHONPATH=${PP}:\$PYTHONPATH && ${CMD}"
