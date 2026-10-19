"""Fig. 1 panels from the finished figure (2026-10-19).

The published Fig. 1 is plots/figure1.pdf, finished by hand in Illustrator (Arial, letters and titles on one line;
the reference for FIGURE_STYLE.md). This cuts it into one file per panel, without letter or title, as
plots/panels/figure1_{a,b,c,d}.{pdf,png}: each panel runs from just below its title line to just above the next
panel's letter. make_figure1.py is the TikZ drawing the Illustrator file started from.

    python split_figure1.py
"""
import os

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, '..', 'plots')


def main():
    src = fitz.open(os.path.join(PLOTS, 'figure1.pdf')); p = src[0]
    H = p.rect.height / 72
    tops = {s['text'].strip(): s['bbox'][1] / 72 for b in p.get_text('dict')['blocks'] for l in b.get('lines', [])
            for s in l['spans'] if round(s['size']) == 15 and s['text'].strip() in 'abcd'}
    ks = sorted(tops, key=tops.get)
    os.makedirs(os.path.join(PLOTS, 'panels'), exist_ok=True)
    p.get_pixmap(dpi=300).save(os.path.join(PLOTS, 'figure1.png'))
    for i, k in enumerate(ks):
        y0 = tops[k] + 0.30
        y1 = (tops[ks[i + 1]] - 0.22) if i + 1 < len(ks) else H
        clip = fitz.Rect(0, y0 * 72, p.rect.width, y1 * 72)
        out = fitz.open(); pg = out.new_page(width=clip.width, height=clip.height)
        pg.show_pdf_page(pg.rect, src, 0, clip=clip)
        out.save(os.path.join(PLOTS, 'panels', 'figure1_%s.pdf' % k))
        pg.get_pixmap(dpi=300).save(os.path.join(PLOTS, 'panels', 'figure1_%s.png' % k))
        print('  figure1 panel %s: %.2f in tall' % (k, y1 - y0))


if __name__ == '__main__':
    main()
