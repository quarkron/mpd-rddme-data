# SI/figS19: `fig:synapse-maps`

**Manuscript file:** tcrpmhc.pdf

**Re-render** (from `scripts/`):

```bash
python tcr_app_fig.py --point ../data/M_react_point.npz --excluded ../data/M_react_excluded.npz --f0-point ../data/F0_react_point.npz --f0-excluded ../data/F0_react_excluded.npz --out ../plots/tcrpmhc
```

**2026-10-09:** moved from main/fig5 to the SI; block a redrawn with the contact zone as a 3-D isometric slab (`tcr_scheme_fig.draw_top_iso`), the edge-on contact and the potential at reduced width beside it, and more space between blocks (figure 8.3 in tall, placed at 0.80 textheight). Re-render writes `../plots/tcrpmhc.{pdf,png}`.

**Plotting:** `tcr_app_fig.py`, `synapse_maps_fig.py`, `tcr_scheme_fig.py`, `synapse_model.py`

**Producing the data:** `./lmrun_bg.sh {0|1} <tag>_<arm> python synapse_run.py --arm {point|excluded} --gauge db --f0 {1|0} --tsim 200 --frames 100 --density 1410 --map-times ... --out out/<tag>_<arm>.npz` (`tcr_pmhc.ipynb` cell 8). M_react was run with map times `0,2,10,30,100,200` and F0_react with `0,2,10,100,200`. Private jLM: `build_jlm.sh`; rate check: `synapse_rate_gate.py`. Logs are in `data/`.

**Data origin (copied into `data/`):** /raid/racda/campaigns/tcr_pmhc/{M_react,F0_react}_{point,excluded}.{npz,log}
