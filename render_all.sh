#!/usr/bin/env bash
# Render every figure of the manuscript (and its per-panel files) from the data in this repository.
#   ./render_all.sh                 # all figures
#   ./render_all.sh main/fig4 SI/figS9   # only these folders
#   AUDIT=1 ./render_all.sh         # also record any file read from outside the repo (tools/external_opens.log)
#   FORCE=1 ./render_all.sh main/fig5   # re-render even a figure saved from Illustrator (overwrites the hand edits)
# Needs: the Python environment of requirements.txt and pdflatex (TeX Live; the figures were made with 2026) on PATH.
# Outputs go to each folder's plots/ (full figure) and plots/panels/ (one file per panel); logs to logs/.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT/common${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
PY=${PYTHON:-python3}
mkdir -p "$ROOT/logs"
[ "${AUDIT:-0}" = 1 ] && : > "$ROOT/tools/external_opens.log"
run () {   # run <folder> <script> [args...]
  local d=$1; shift
  if [ $# -gt 0 ] && [ -n "${SEL:-}" ] && ! [[ " $SEL " == *" $d "* ]]; then return; fi
  local log="$ROOT/logs/$(echo $d | tr / _).log"
  # a figure finished by hand in Illustrator is not overwritten: skip it unless FORCE=1
  if [ "${FORCE:-0}" != 1 ] && $PY "$ROOT/tools/illustrator_edited.py" "$d" > "$log" 2>&1; then
    printf '%-12s %-28s SKIPPED: edited in Illustrator (%s); FORCE=1 to re-render\n' "$d" "$1" "$(tr -d '\n' < "$log" | sed 's/^ *//')"
    return
  fi
  if [ "${AUDIT:-0}" = 1 ]; then
    (cd "$ROOT/$d/scripts" && $PY "$ROOT/tools/audit_open.py" "$@") > "$log" 2>&1
  else
    (cd "$ROOT/$d/scripts" && $PY "$@") > "$log" 2>&1
  fi
  local rc=$?; printf '%-12s %-28s rc=%d\n' "$d" "$1" $rc; [ $rc = 0 ] || FAIL="$FAIL $d"
}
SEL="$*"; FAIL=""
run main/fig1  split_figure1.py     # the published Fig. 1 is the Illustrator file plots/figure1.pdf
run main/fig2  rxthree_relax_fig.py
run main/fig3  rx_field_fig.py
run main/fig4  fig_phase.py
run main/fig5  kip3_app_fig.py --pt ../data/oct09b/c32_pt_iso_L1000_s1.npz --exc ../data/oct09b/c32_exc_iso_L1000_s1.npz \
               --sw ../data/oct09b/c32_exc_sw_L1000_s1.npz --out ../plots/kip3_traffic
run SI/figS1   wm_sim_figs.py
run SI/figS2   wm_closedform_figs.py
run SI/figS3   wm6_plot.py
run SI/figS4   wm_closedform_figs.py
run SI/figS4   wm_sim_figs.py
run SI/figS8   fig_polymer_r2.py
run SI/figS9   lp_match_fig.py
run SI/figS10  cpu_gpu_validation_fig.py
run SI/figS11  adv_fig.py
run SI/figS12  two4_readdy_fig.py
run SI/figS13  rxthree_kl_fig.py
run SI/figS14  rxthree_rho_fig.py
run SI/figS15  rx_K_main_fig.py
run SI/figS16  rx_field_counts_fig.py
run SI/figS17  rx_field_kl_fig.py
run SI/figS18  tau_ladder_fig.py
run SI/figS19  demix_starts_fig.py
run SI/figS20  tcr_app_fig.py --point ../data/M_react_point.npz --excluded ../data/M_react_excluded.npz \
               --f0-point ../data/F0_react_point.npz --f0-excluded ../data/F0_react_excluded.npz --out ../plots/tcrpmhc
# TikZ figures (S5-S7): the drawings are in the SI's .tex; this writes each panel on its own
for spec in "SI/figS5 si_fccmoves a b" "SI/figS6 si_apex a b" "SI/figS7 si_growth N3 N8 N20 N60"; do
  set -- $spec; d=$1; n=$2; shift 2
  if [ -n "$SEL" ] && ! [[ " $SEL " == *" $d "* ]]; then continue; fi
  (cd "$ROOT" && $PY common/tikz_panels.py $d/scripts/$n.tex $n "$@") > "$ROOT/logs/$(echo $d | tr / _).log" 2>&1
  rc=$?; printf '%-12s %-28s rc=%d\n' "$d" "tikz_panels $n" $rc; [ $rc = 0 ] || FAIL="$FAIL $d"
done
$PY "$ROOT/tools/make_ai.py"          # an Illustrator .ai beside every figure and panel PDF
[ -z "$FAIL" ] && echo "ALL FIGURES RENDERED" || { echo "FAILED:$FAIL (see logs/)"; exit 1; }
