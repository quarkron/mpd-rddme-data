# VEX-RDDME manuscript: figures, data and scripts

Contianeed in this repo are the figures, raw data, and scripts used to generate the figures for the manuscript.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./render_all.sh                     # every figure; or e.g. ./render_all.sh main/fig4 SI/figS8
```

Requirements:

- **Python 3.12** with the packages in `requirements.txt` (numpy, scipy, matplotlib, h5py, Pillow, PyMuPDF; the
  versions the published figures were made with).
- **pdflatex on `PATH`.** Matplotlib sets all text through LaTeX (`text.usetex`), and Fig. 1 and the SI drawings
  S5-S7 are TikZ. Needed packages: `amsmath`, `amssymb`, `bm`, `sansmath`, `tikz`, `standalone`, `type1cm`
  (matplotlib's own). The figures were made with TeX Live 2026; any recent full TeX Live works.

`render_all.sh` runs each folder's plotting script from its `scripts/` folder with `PYTHONPATH=common`, writes the full
figure to `<folder>/plots/` and one file per panel to `<folder>/plots/panels/`, and logs to `logs/`. It prints one line
per figure and ends with `ALL FIGURES RENDERED`. Rendering everything takes about 20 minutes on one core.

To render a single figure by hand:

```bash
export PYTHONPATH=$PWD/common
cd main/fig4/scripts && python fig_polymer_r2.py         # -> main/fig4/plots/polymer_r2.{pdf,png} and plots/panels/
```

The plotting scripts never write outside the repository. Several accept `--install`, which copies the result into a
manuscript checkout's `figures/` folder; set `MANUSCRIPT_FIGURES=/path/to/manuscript/figures` to use it.

## Layout

```
common/                  modules shared by several figures
  jcp_style.py             figure style (sizes, fonts, block headlines), width and overlap checks
  panel_export.py          writes each lettered panel of a figure as its own file, without letter or headline
  tikz_panels.py           the same for the TikZ figures S5-S7
  kip3_palette.py, fcc_exact.py, chain_geometry.py, fixed_chain_harness.py
main/figN/, main/tableI/ the main-text figures and Table I
SI/figSN/, SI/tableV/    the Supplementary figures and Table V
  README.md                manuscript label, the render command, how the data were produced, known gaps
  scripts/                 the plotting script and its local modules, plus the simulation, campaign and analysis
                           drivers that made the data (archived as they were run; see "Regenerating the data")
  data/                    the simulation output the figure reads, with the run logs
  plots/                   the rendered figure; plots/panels/ one file per panel
render_all.sh            renders everything (AUDIT=1 also checks that nothing is read from outside the repo)
tools/audit_open.py      the read audit used by AUDIT=1
requirements.txt
```

## Figures

| Folder | Manuscript | Label | Plotting script | Data |
|---|---|---|---|---|
| main/fig1 | Fig. 1 | fig:schematic | make_figure1.py (TikZ) | none |
| main/fig2 | Fig. 2 | fig:three-relax-readdy | rxthree_relax_fig.py | 232 MB |
| main/fig3 | Fig. 3 | fig:rx-field-kl | rx_field_fig.py | 15 MB |
| main/fig4 | Fig. 4 | fig:polymer | fig_polymer_r2.py | 112 MB |
| main/fig5 | Fig. 5 | fig:phase | fig_phase.py | main/fig4/data |
| main/fig6 | Fig. 6 | fig:kip3-traffic | kip3_app_fig.py (arguments in render_all.sh) | 49 MB |
| main/tableI | Table I | tab:cost | none, assembled by hand (README maps every entry to its source) | 0.3 MB |
| SI/figS1 | Fig. S1 | fig:wm-convergence | wm_sim_figs.py | 32 MB |
| SI/figS2 | Fig. S2 | fig:wm-flatness-parameters | wm_closedform_figs.py | closed form |
| SI/figS3 | Fig. S3 | fig:wm-omega | wm6_plot.py | 23 MB |
| SI/figS4 | Fig. S4 | fig:wm-intravoxel | wm_closedform_figs.py (a), wm_sim_figs.py (b) | SI/figS1/data |
| SI/figS5 | Fig. S5 | fig:si-fccmoves | TikZ in scripts/si_fccmoves.tex | none |
| SI/figS6 | Fig. S6 | fig:si-apex | TikZ in scripts/si_apex.tex | none |
| SI/figS7 | Fig. S7 | fig:si-growth | TikZ in scripts/si_growth.tex | none |
| SI/figS8 | Fig. S8 | fig:lp-match | lp_match_fig.py | 32 kB (+ main/fig4/data) |
| SI/figS9 | Fig. S9 | fig:cpu-gpu-validation | cpu_gpu_validation_fig.py | 0.9 MB |
| SI/figS10 | Fig. S10 | fig:advantage | adv_fig.py | 0.3 MB |
| SI/tableV | Table V | tab:advantage | adv_table.py | 0.2 MB |
| SI/figS11 | Fig. S11 | fig:two-eq-readdy | two4_readdy_fig.py | 1.8 MB |
| SI/figS12 | Fig. S12 | fig:three-kl | rxthree_kl_fig.py | main/fig2/data |
| SI/figS13 | Fig. S13 | fig:three-rho | rxthree_rho_fig.py | main/fig2/data |
| SI/figS14 | Fig. S14 | fig:rx-K-main | rx_K_main_fig.py | 0.9 MB |
| SI/figS15 | Fig. S15 | fig:rx-field-counts | rx_field_counts_fig.py | main/fig3/data |
| SI/figS16 | Fig. S16 | fig:rx-field-kl-time | rx_field_kl_fig.py | main/fig3/data |
| SI/figS17 | Fig. S17 | fig:tau-ladder | tau_ladder_fig.py | 243 MB |
| SI/figS18 | Fig. S18 | fig:demix-starts | demix_starts_fig.py | main/fig4/data |
| SI/figS19 | Fig. S19 | fig:synapse-maps | tcr_app_fig.py (arguments in render_all.sh) | 0.7 MB |

Figures that use another figure's data read it in place through a relative path (for example S12 and S13 read
`../../../main/fig2/data`), so each data set is stored once. Fig. 5 and S18 also import their drawing code from
`main/fig4/scripts`.

**Panels.** Every figure also writes its panels as separate files, `plots/panels/<name>_<letter>.{pdf,png}`, without
the letter or headline. Unlettered two-panel figures (S12, S15, S16) are split into `_left` and `_right`; a piece shared
by two panels (S11's key, S19's colour bar) goes to `_shared`; single-panel figures (S1, S13, S14) write none. S5-S7
are split by `common/tikz_panels.py` (called from `render_all.sh`); S4's two panels are its two files.

## Regenerating the data

The data in `data/` are the outputs of simulations that are expensive and need software beyond this repository:

- **MPD-RDDME** (the lattice method of the paper): Lattice Microbes 2.6 with the MPD-RDDME extensions,
  `github.com/luthey-schulten-chemistry/Lattice_Microbes_2.6`, run on NVIDIA GPUs in the `lm2.6` container. The
  microtubule runs of Fig. 6 used commit `9466763`; each folder's README names the build its runs used.
- **ReaDDy 2.0.13** (the off-lattice reference) in a conda environment with Python 3.9.
- Closed-form references (S2, S4a, the exact fcc chain statistics of Fig. 4 and S8) are computed by the plotting
  scripts and `common/fcc_exact.py`; they need nothing else.

Each figure's `scripts/` holds the drivers exactly as they were run (`*_campaign.sh`, `*_run.py`, `lmrun.sh`, ...),
and its README gives the command lines, seeds and run lengths. The drivers still contain the paths of the cluster they
ran on (`/lm`, `/ftsz/studies/...`, `/raid/...`): point those at your own engine build and output folder before running
them. Plotting does not depend on any of them.

Not included, because the figures do not read them: the raw per-run campaign behind S17 (1 GB; `SI/figS17/data` holds
the per-run outputs the figure uses) and the step-by-step logs of the S8 persistence-length scan (1 GB; the scan
results are `SI/figS8/data/lp_scan_n{11,32}.npz`).

## Known provenance gaps

- Fig. 1 was drawn for the manuscript in TikZ (`make_figure1.py`); `main/fig1/plots/published/` keeps the original,
  earlier print for reference.
- S5-S7 are TikZ written directly in the SI source; no generator exists beyond the `.tex` kept here.
- Fig. 3, S8, S10, S12, S13: some runs have no saved launcher; the commands in their READMEs were rebuilt from the run
  metadata and are marked as such.
- Fig. 4: the demixing runs and the ReaDDy bath runs have no recorded invocation; several inputs were produced by
  scripts with uncommitted edits at run time, and the copies here are those working-tree versions.
- S3: the original driver lived in a temporary folder (copied here); `parse_reps.py` hard-codes that stale path.
- Table I is assembled by hand; `main/tableI/README.md` maps each entry to its source file.
