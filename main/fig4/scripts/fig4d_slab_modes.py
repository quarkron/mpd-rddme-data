"""EXPERIMENT (2026-10-09): the slab-start block d in both colourings, side by side (demix_slab_maps.py)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fig_polymer_r2 as F                      # noqa: E402
import demix_slab_maps as M                     # noqa: E402
import matplotlib.pyplot as plt                 # noqa: E402
ROOT = os.path.join(HERE, '..', 'data', 'demix_noneq')
rc, _ = F.style()
with plt.rc_context(rc):
    for kind in ('slab', 'droplet'):
        fig = plt.figure(figsize=(7.4, 2.6)); W = 0.95
        for g, mode in enumerate(('onetone', 'twotone')):
            x0 = 0.45 + g * 3.75
            grid = [[fig.add_axes([(x0 + c * 1.0) / 7.4, (1.32 - r * 1.05) / 2.6, W / 7.4, W / 2.6]) for c in range(3)] for r in range(2)]
            cax = fig.add_axes([(x0 + 3.0 + 0.02) / 7.4, 0.27 / 2.6, 0.06 / 7.4, 2.0 / 2.6])
            M.panel_maps(grid, cax, os.path.join(ROOT, kind), F.ARMS_D if g == 0 else [(r, '') for r, _ in F.ARMS_D], (0, 1, 10), mode)
        out = os.path.join(HERE, '..', 'plots', 'experiments', 'fig4d_%s_modes' % kind)
        for e in ('pdf', 'png'): fig.savefig(out + '.' + e, dpi=250)
        print('wrote', out)
