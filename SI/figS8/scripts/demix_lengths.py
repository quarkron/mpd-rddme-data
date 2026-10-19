"""Chain-length spread for the demixing block of fig_polymer_r2: how long the assembling polymers are, over time.

Two panels, linear axes, both arms of block d on each:

  left   the median chain length against time, with the 1st-99th percentile band of the population
  right  the length distribution at the end of the run, as the fraction of chains at each length

★THIS IS THE POPULATION'S SPREAD, NOT INDIVIDUAL CHAINS FOLLOWED IN TIME. The recorder stores a histogram of chain
lengths per frame (len_hist, every 0.25 s); a chain's identity cannot be followed from frame to frame because the
topology commit renumbers complexes at every chemical event (see gradient_hook.py, --track-bodies). A band here is
therefore the distribution across chains at each time, which is what "spread" can honestly mean for this record.

★The histogram is clipped at nx = 64 subunits (gradient_hook.py: bincount(...)[:nx + 1]); a chain longer than that
is dropped from its frame. That is 1-3 chains of ~2500 in these runs, so the quantiles are unaffected, and the count
of dropped chains is printed rather than hidden.
"""
import os

import numpy as np

QUANTILES = (0.01, 0.50, 0.99)


def length_course(path):
    """times, quantiles (n_frames, 5) of chain length in subunits, n_chains, and the final histogram."""
    d = np.load(path, allow_pickle=True)
    h = np.asarray(d['len_hist'], float)
    t = np.asarray(d['times'], float)
    L = np.arange(h.shape[1])
    q = np.full((len(t), len(QUANTILES)), np.nan)
    for i, row in enumerate(h):
        n = row.sum()
        if n > 0:
            c = np.cumsum(row) / n
            q[i] = [L[np.searchsorted(c, p)] for p in QUANTILES]
    lost = int(np.asarray(d['n_chains'])[-1] - h[-1].sum())
    return t, q, h.sum(axis=1), h[-1], lost


def panel_lengths(ax_t, ax_h, arms, demix_dir, colours, styles, note=True):
    """arms: ((run, label), ...) as in block d; colours and line styles match its variance panel."""
    for (run, lab), col, ls in zip(arms, colours, styles):
        t, q, n, h_end, lost = length_course(os.path.join(demix_dir, run + '.npz'))
        ok = n > 0
        # The median and the 1st-99th percentile band: the bulk and the whole spread, nothing between.
        ax_t.fill_between(t[ok], q[ok, 0], q[ok, 2], color=col, alpha=0.18, lw=0)
        ax_t.plot(t[ok], q[ok, 1], ls, color=col, lw=1.5, label=lab)
        L = np.arange(len(h_end))
        ax_h.step(L, h_end / max(h_end.sum(), 1), ls, where='mid', color=col, lw=1.2, label=lab)
        mean_end = float((L * h_end).sum() / max(h_end.sum(), 1))
        print('  %-18s chain length at t = %.0f s: median %d, 1-99 %% %d-%d, mean %.1f, %d chains '
              '(%d beyond the %d-subunit histogram)'
              % (run, t[-1], q[-1, 1], q[-1, 0], q[-1, 2], mean_end, int(n[-1]), lost, len(h_end) - 1))
    ax_t.set_xlabel('time (s)')
    ax_t.set_ylabel('chain length (subunits)')
    ax_t.set_xlim(0, 10); ax_t.set_ylim(0, 72)
    ax_t.grid(alpha=0.25, lw=0.4)
    # ★No percent sign in figure text: under usetex it opens a TeX comment and silently drops the rest.
    if note:
        ax_t.text(0.04, 0.965, 'line: median;  band: 1st to 99th percentile',
                  transform=ax_t.transAxes, ha='left', va='top', fontsize=6.3, color='0.30')
    ax_h.set_xlabel('chain length at $t = 10$ s (subunits)')
    ax_h.set_ylabel('fraction of chains')
    ax_h.set_xlim(0, 64); ax_h.set_ylim(bottom=0)
    ax_h.grid(alpha=0.25, lw=0.4)
    ax_h.legend(loc='upper right', fontsize=7, frameon=False, handlelength=1.8, borderpad=0.1)
