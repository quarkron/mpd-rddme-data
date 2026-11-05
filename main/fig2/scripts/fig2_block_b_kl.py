"""Fig. 2 block b = the divergence of the occupancy distributions from BD over the relaxation (2026-11-05).

The owner swapped Fig. 2b (the occupancy distributions at tau 20) with SI Fig. S13 (this divergence): the distributions
are now SI/figS13 (rxthree_eq_fig.py). Fig. 2 is the owner's Illustrator file, so it is not re-rendered: this script
draws block b alone on the same 8.5 x 11 in page, at the old block's axes (rxthree_relax_fig.py e0/e1), removes the old
block from the Illustrator page and places the new one 1:1. Every other part of the page is checked unchanged.

    python fig2_block_b_kl.py

Reads ../plots/pdf/rxthree_relax_prekl.pdf (the Illustrator figure before the swap), writes ../plots/pdf/rxthree_relax.pdf
(+ ../plots/png/rxthree_relax.png) and ../plots/pdf/rxthree_relax_block_b.pdf (the block alone). Prints the medians the
main text quotes.
"""
import os, sys

import numpy as np
import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import matplotlib                                            # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
import jcp_style as J                                        # noqa: E402
import rxthree_data as RD                                    # noqa: E402

PLOTS = os.path.join(HERE, '..', 'plots')
SRC = os.path.join(PLOTS, 'pdf', 'rxthree_relax_prekl.pdf')
OUT = os.path.join(PLOTS, 'pdf', 'rxthree_relax.pdf')
BLOCK = os.path.join(PLOTS, 'pdf', 'rxthree_relax_block_b.pdf')

XR, WR, YB = 3.55, J.PAGE_W - J.MARGIN - 3.55, J.MARGIN     # block b, as in rxthree_relax_fig.py
AX = [(XR + 0.55, 1.35, 1.58, 1.70), (XR + 2.75, 1.35, 1.58, 1.70)]
TITLE = 'MPD-RDDME occupancy distributions match BD\nmore closely than point particles'
# the old block on the page, in pt from the top-left: title line, then the two plots (letter b and blocks a, c kept)
CLEAR = [fitz.Rect(272, 33, 590, 76), fitz.Rect(247, 76, 590, 256)]


def draw_block():
    D = RD.prepare()
    grid, L, arms = D['grid'], D['L'], D['arms']
    xt = [0, 0.1, 1, 10]
    with plt.rc_context(J.PUB_RC):
        fig = J.pub_page()
        fig.patch.set_alpha(0)
        # as rxthree_relax_fig.header(): first line on the letter's baseline, centred right of the letter
        lt, tt = J.pub_header(fig, 'b', TITLE, XR, YB, WR)
        lt.remove()                                          # the page keeps its own letter
        tt.set_multialignment('center')
        tt.set_y(tt.get_position()[1] - TITLE.count('\n') * 1.2 * J.TITLE_PT / 72.0 / J.PAGE_H)
        tt.set_x((XR + 0.30 + 0.5 * (WR - 0.30)) / J.PAGE_W)
        for (x, yt, w, h), kk, name in zip(AX, ('klA', 'klB'), 'AB'):
            a = J.pub_ax(fig, x, yt, w, h)
            for key, lbl, col, sty in arms:
                yv = np.asarray(L[key][kk])
                a.plot(grid, yv, sty, color=col, mfc=col, mew=0, ms=3.0, lw=1.2, label=lbl)
                print('  %s %-16s median tau>=1 %.3g; peak %.3g at tau %.3g; tau 0 %.3g'
                      % (kk, lbl, np.median(yv[grid >= 1.0]), yv.max(), grid[np.argmax(yv)], yv[0]))
            a.set_xscale('symlog', linthresh=grid[1], linscale=0.5); a.set_yscale('log')
            a.set_xlim(-0.02, grid[-1] * 1.15)
            a.set_xticks(xt); a.set_xticklabels(['%g' % v for v in xt])
            a.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            a.tick_params(which='both', top=True, right=True)
            a.set_xlabel(r'time $\tau$')
            a.set_ylabel(r'$D_{\mathrm{KL}}$ of $n_\mathrm{%s}$ from BD' % name)
            a.grid(alpha=0.25, lw=0.4, which='both')
            a.set_ylim(1e-4, 5.0)
            a.legend(loc='lower left', fontsize=J.TEXT_PT, frameon=False, handlelength=1.8, labelspacing=0.25,
                     borderpad=0.1, handletextpad=0.4, borderaxespad=0.2)
        fig.savefig(BLOCK, transparent=True)
        plt.close(fig)
    J.pub_fonts(BLOCK, 'rxthree_relax block b')


def place_block():
    doc = fitz.open(SRC); page = doc[0]
    before = page.get_pixmap(dpi=100)
    for r in CLEAR:
        page.add_redact_annot(r, fill=False)
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                          text=fitz.PDF_REDACT_TEXT_REMOVE)
    page.show_pdf_page(page.rect, fitz.open(BLOCK), 0)
    # Illustrator reopens from its private copy (PieceInfo) and would show the old block: drop it (as for Figs. 1, 5)
    doc.xref_set_key(page.xref, 'PieceInfo', 'null')
    doc.save(OUT, garbage=3, deflate=True)
    # nothing outside the cleared rectangles moved
    after = fitz.open(OUT)[0].get_pixmap(dpi=100)
    W, s = before.width, 100 / 72.0
    a, b, n = before.samples, after.samples, before.n
    bad = 0
    for yy in range(before.height):
        for xx in range(0, W, 2):
            px, py = xx / s, yy / s
            if any(r.contains(fitz.Point(px, py)) for r in CLEAR):
                continue
            i = (yy * W + xx) * n
            if max(abs(a[i + c] - b[i + c]) for c in range(3)) > 8:
                bad += 1
    print('  pixels changed outside block b: %d' % bad)
    fitz.open(OUT)[0].get_pixmap(dpi=300).save(os.path.join(PLOTS, 'png', 'rxthree_relax.png'))
    print('wrote', os.path.normpath(OUT))


if __name__ == '__main__':
    draw_block()
    place_block()
