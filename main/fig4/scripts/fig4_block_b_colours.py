"""Fig. 4 block b recoloured (2026-11-05): finite-size in ink, point particles in Okabe-Ito orange.

Navy and green mean crowder and polymer in blocks a and c, so block b no longer uses them for the two arms. Fig. 4 is
the owner's Illustrator file, so it is not re-rendered: this swaps the colour operators of block b's lines (page
stream, after block a) and of its two percentile bands (their own form XObjects), drops PieceInfo and checks that
nothing outside block b moved.

    python fig4_block_b_colours.py

Reads ../plots/phase_precolour.pdf (the Illustrator figure before the change), writes ../plots/phase.{pdf,png}.
"""
import os

import fitz

PLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plots')
SRC, OUT = os.path.join(PLOTS, 'phase_precolour.pdf'), os.path.join(PLOTS, 'phase.pdf')
NAVY, GREEN = b'0.071 0.247 0.478', b'0.612 0.749 0.235'
INK, ORANGE = b'0.106 0.141 0.188', b'0.902 0.624 0'      # jcp_style.INK, Okabe-Ito #E69F00
BLOCK_B = fitz.Rect(390, 95, 590, 350)                   # pt from the top-left


def recolour(b):
    return b.replace(NAVY, INK).replace(GREEN, ORANGE)


def main():
    doc = fitz.open(SRC); page = doc[0]
    before = page.get_pixmap(dpi=100)
    n = 0
    for x in range(1, doc.xref_length()):                # the bands: forms whose BBox lies in block b
        if doc.xref_get_key(x, 'Subtype')[1] != '/Form' or not doc.xref_stream(x):
            continue
        x0, y1, x1, y0 = map(float, doc.xref_get_key(x, 'BBox')[1].strip('[]').split())
        if fitz.Rect(x0, 792 - max(y0, y1), x1, 792 - min(y0, y1)).intersects(BLOCK_B):
            st = doc.xref_stream(x)
            if NAVY in st or GREEN in st:
                doc.update_stream(x, recolour(st)); n += 1
    (cx,) = page.get_contents()
    st = doc.xref_stream(cx)
    cut = st.index(b'0.106 0.141 0.188 RG')               # block b's axes come after every navy/green of block a
    doc.update_stream(cx, st[:cut] + recolour(st[cut:st.index(b'/Im0 Do')]) + st[st.index(b'/Im0 Do'):])
    doc.xref_set_key(page.xref, 'PieceInfo', 'null')      # Illustrator would reopen its private copy otherwise
    doc.save(OUT, garbage=3, deflate=True)
    after = fitz.open(OUT)[0].get_pixmap(dpi=100)
    s, W, k = 100 / 72.0, before.width, before.n
    bad = sum(1 for yy in range(before.height) for xx in range(0, W, 2)
              if not BLOCK_B.contains(fitz.Point(xx / s, yy / s))
              and max(abs(before.samples[(yy * W + xx) * k + c] - after.samples[(yy * W + xx) * k + c])
                      for c in range(3)) > 8)
    print('  %d bands recoloured; pixels changed outside block b: %d' % (n, bad))
    fitz.open(OUT)[0].get_pixmap(dpi=300).save(os.path.join(PLOTS, 'phase.png'))
    print('wrote', os.path.normpath(OUT))


if __name__ == '__main__':
    main()
