"""Rewrite low-bit palette images in PDFs as 8-bit RGB, pixel for pixel (2026-10-19).

    python tools/fix_palette_images.py file.pdf [...]

Matplotlib stores an image with 16 colors or fewer as a 1-, 2- or 4-bit indexed (palette) image. Adobe Illustrator
misreads those and fills the whole image with one color (the point-particle kymograph of Fig. 5 became solid cyan).
This replaces each such image with the same pixels as an 8-bit RGB image; nothing else in the file changes.
common/jcp_style.py now makes matplotlib write RGB in the first place, so this is only for existing files.
"""
import os
import re
import sys

import fitz


def fix(path):
    d = fitz.open(path); n = 0
    for page in d:
        for img in page.get_images(full=True):
            xref = img[0]
            o = d.xref_object(xref, compressed=True)
            m = re.search(r'/BitsPerComponent (\d+)', o)
            if '/Indexed' not in o or not m or int(m.group(1)) >= 8:
                continue
            pix = fitz.Pixmap(d, xref)
            if pix.colorspace is None or pix.colorspace.n != 3 or pix.alpha:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            page.replace_image(xref, pixmap=pix)
            n += 1
    if n:
        tmp = path + '.fixtmp'
        d.save(tmp, garbage=3, deflate=True); d.close(); os.replace(tmp, path)
    print('  %s: %d palette image(s) rewritten as RGB' % (path, n))
    return n


if __name__ == '__main__':
    for p in sys.argv[1:]:
        fix(p)
