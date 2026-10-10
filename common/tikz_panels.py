"""Split a TikZ figure (a figure environment holding one tikzpicture per panel) into one PDF/PNG per panel, without
the panel's label line or the caption (2026-10-09; the TikZ half of panel_export.py).

    python tikz_panels.py SI/figS5/scripts/si_fccmoves.tex si_fccmoves a b
    python tikz_panels.py SI/figS7/scripts/si_growth.tex si_growth N3 N8 N20 N60

Each tikzpicture, in order, is set alone in a standalone document and written to <figure folder>/plots/panels/
<name>_<key>.pdf, with a 300 dpi PNG beside it. Run from studies/vex-rddme with TeX Live 2026 on PATH.
"""
import os
import shutil
import re
import subprocess
import sys
import tempfile

PRE = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb,bm}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,decorations.pathmorphing,patterns}
\begin{document}
%s
\end{document}
"""


def main():
    src, name, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
    tex = open(src).read()
    pics = re.findall(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', tex, flags=re.S)
    if len(pics) != len(keys):
        sys.exit('%s: %d tikzpictures, %d keys given' % (src, len(pics), len(keys)))
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(src)), '..', 'plots', 'panels'))
    os.makedirs(out, exist_ok=True)
    import fitz
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        for k, pic in zip(keys, pics):
            stem = '%s_%s' % (name, k)
            open(os.path.join(tmp, stem + '.tex'), 'w').write(PRE % pic)
            r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex'], cwd=tmp,
                               capture_output=True, text=True)
            if r.returncode:
                sys.exit('%s failed:\n%s' % (stem, r.stdout[-1500:]))
            pdf = os.path.join(out, stem + '.pdf')
            shutil.move(os.path.join(tmp, stem + '.pdf'), pdf)
            d = fitz.open(pdf)
            d[0].get_pixmap(dpi=300).save(os.path.join(out, stem + '.png'))
            w, h = d[0].rect.width / 72, d[0].rect.height / 72
            print('  %s: %.2f x %.2f in' % (stem, w, h))
    print('  panels -> %s' % out)


if __name__ == '__main__':
    main()
