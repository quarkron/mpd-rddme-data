#!/usr/bin/env bash
# Rebuild the private jLM used by this study. The container's own jLM ships Lattice.pyx
# uncompiled, and /lm is not writable as this uid, so the copy lives in the campaign tree.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D=/raid/racda/campaigns/tcr_pmhc/jlm
rm -rf "$D"; mkdir -p "$D"
cp -r "${REPO}/Lattice_Microbes_2.6/src/jlm/jLM" "$D/"
rm -rf "$D/jLM/__pycache__"
docker run --rm -v /raid/racda/campaigns:/raid/racda/campaigns -w "$D/jLM" \
    -e USER="$(id -un)" -e HOME=/tmp -u "$(id -u):$(id -g)" lm2.6:latest \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && \
             export CFLAGS=\"-I\$(python -c 'import numpy;print(numpy.get_include())')\" && \
             cythonize -i Lattice.pyx"
ls -la "$D/jLM"/Lattice*.so
