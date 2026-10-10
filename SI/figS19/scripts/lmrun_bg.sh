#!/usr/bin/env bash
# Launch a run as a DETACHED docker container, so its lifetime does not depend on any shell.
# A `setsid nohup ./lmrun.sh ... &` launch was killed mid-run at 45% with no error in the log;
# `docker run -d` survives because the daemon owns the container, not the calling shell.
#   ./lmrun_bg.sh <gpu> <name> python synapse_run.py ... --out out/foo.npz
# The container writes its own log to out/<name>.log. Poll with `docker ps` / `docker logs`.
set -euo pipefail
GPU="$1"; NAME="$2"; shift 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LM="${REPO}/Lattice_Microbes_2.6"
HERE="/ftsz/studies/tcr-pmhc"
CAMP="/raid/racda/campaigns/tcr_pmhc"
JLM="${CAMP}/jlm"
PP="${JLM}:/lm/build_container:/lm/build_container/pylm:/ftsz/studies/common"
CMD="$(printf "%q " "$@")"   # preserve per-argument quoting through bash -c
docker rm -f "syn_${NAME}" >/dev/null 2>&1 || true
docker run -d --name "syn_${NAME}" --gpus "\"device=${GPU}\"" \
    -v "${LM}":/lm -v "${REPO}":/ftsz -v /raid/racda/campaigns:/raid/racda/campaigns -v /raid/racda/opt:/raid/racda/opt:ro -v /raid/racda/tmp:/raid/racda/tmp \
    -w "${HERE}" -u "$(id -u):$(id -g)" -e USER="$(id -un)" -e HOME=/tmp \
    lm2.6:latest \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && \
             export PATH=/raid/racda/opt/texlive/2026/bin/x86_64-linux:\$PATH && export TMPDIR=/raid/racda/tmp && export PYTHONPATH=${PP}:\$PYTHONPATH && ${CMD} > ${CAMP}/${NAME}.log 2>&1"
echo "launched syn_${NAME} on gpu ${GPU}"
