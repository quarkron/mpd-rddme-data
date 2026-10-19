"""Split a TikZ figure (a figure environment holding one tikzpicture per panel) into one PDF/PNG per panel, without
the panel's label line or the caption, and write the combined figure with bold panel letters (2026-10-09; the TikZ
half of panel_export.py; type and combined file 2026-10-19).

    python tikz_panels.py SI/figS5/scripts/si_fccmoves.tex si_fccmoves a b
    python tikz_panels.py SI/figS7/scripts/si_growth.tex si_growth N3 N8 N20 N60

Each tikzpicture, in order, is set alone in a standalone document and written to <figure folder>/plots/panels/
<name>_<key>.pdf, with a 300 dpi PNG beside it. The combined figure, <figure folder>/plots/<name>.{pdf,png}, sets
every tikzpicture side by side as the figure's tabular does, each with its bold 15 pt letter (a, b, ...) at its
top-left, all letters on one baseline; the caption is left out, and so is a label row of '(a) ...' descriptions
(those belong to the caption), while a row of short labels (S7's N = 3, 8, 20, 60) is kept under the drawings.
Run from studies/vex-rddme with TeX Live 2026 on PATH.

TYPE (FIGURE_STYLE.md sections 1 and 5): TeX Gyre Heros (tgheros, the Helvetica clone TeX Live ships; the URW
helvet metrics are not installed) for text, and for math through sansmath, so no Computer Modern is embedded.
The base size is 12 pt, so \\small is 10.95 pt and \\footnotesize 10 pt; \\scriptsize and \\tiny are raised to
10 pt, so nothing in a drawing is set below 10 pt. The one math glyph sansmath still takes from Computer Modern,
the degree sign (^\\circ), is replaced by the text degree sign before compiling.
"""
import os
import shutil
import re
import subprocess
import sys
import tempfile

PRE = r"""\documentclass[12pt,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{tgheros}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{amsmath,amssymb,bm}
\usepackage{sansmath}
\usepackage{textcomp}
\renewcommand{\scriptsize}{\fontsize{10}{12}\selectfont}
\renewcommand{\tiny}{\fontsize{10}{12}\selectfont}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,decorations.pathmorphing,patterns}
\begin{document}
\sansmath
%s
\end{document}
"""
LABEL = r'\node[anchor=north,inner sep=0pt] at ([yshift=-5pt]current bounding box.south) {%s};'
LETTER = (r'\node[anchor=base west,inner sep=0pt,font=\fontsize{15}{18}\selectfont\bfseries] (pnlletter) at '
          r'([xshift=-2pt,yshift=8pt]current bounding box.north west) {%s};')


def sans(pic):
    """The few math glyphs sansmath leaves in Computer Modern, written in text."""
    return re.sub(r'\$([^$]*?)\^\\circ\$', r'$\1$\\textdegree{}', pic)


def compile_pdf(tmp, stem, body, pdf):
    open(os.path.join(tmp, stem + '.tex'), 'w').write(PRE % body)
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex'], cwd=tmp,
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit('%s failed:\n%s' % (stem, r.stdout[-1500:]))
    shutil.move(os.path.join(tmp, stem + '.pdf'), pdf)
    import fitz
    d = fitz.open(pdf)
    d[0].get_pixmap(dpi=300).save(pdf[:-4] + '.png')
    fonts = sorted({f[3].split('+')[-1] for f in d[0].get_fonts()})
    bad = [f for f in fonts if not f.startswith('TeXGyreHeros')]
    print('  %s: %.2f x %.2f in, fonts %s%s' % (stem, d[0].rect.width / 72, d[0].rect.height / 72, ', '.join(fonts),
                                                ('  << NOT SANS: %s' % bad) if bad else ''))


def combined(tex, pics):
    """The figure's tabular with a letter on each drawing and without the caption or '(a) ...' label row; a row of
    short labels (S7's N = 3, ...) goes into each drawing, centred just under it."""
    spec = re.search(r'\\begin\{tabular\}\{(.*?)\}\s*\n', tex).group(1)
    labels = [None] * len(pics)
    m = re.search(r'\\end\{tikzpicture\}\s*\\\\(\[[^\]]*\])?\s*\n(.*?)\\\\\s*\n\s*\\end\{tabular\}', tex, flags=re.S)
    if m:
        cells = [c.strip() for c in m.group(2).split('&')]
        if len(cells) == len(pics) and not any(re.match(r'\{?(\\\w+\s*)?\([a-h]\)', c) for c in cells):
            labels = cells
    lettered = []
    for i, pic in enumerate(pics):
        body = re.sub(r'\\begin\{tikzpicture\}(\[[^\]]*\])?', lambda _: r'\begin{tikzpicture}[baseline=(pnlletter.base)]',
                      pic, count=1)
        add = ''
        if labels[i]:
            add += LABEL % labels[i] + '\n'
        add += LETTER % chr(ord('a') + i) + '\n'
        lettered.append(body.replace(r'\end{tikzpicture}', add + r'\end{tikzpicture}'))
    return '\\begin{tabular}{%s}\n%s\n\\end{tabular}' % (spec, ' &\n'.join(lettered))


def main():
    src, name, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
    tex = open(src).read()
    pics = [sans(p) for p in re.findall(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', tex, flags=re.S)]
    if len(pics) != len(keys):
        sys.exit('%s: %d tikzpictures, %d keys given' % (src, len(pics), len(keys)))
    plots = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(src)), '..', 'plots'))
    out = os.path.join(plots, 'panels')
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        for k, pic in zip(keys, pics):
            stem = '%s_%s' % (name, k)
            compile_pdf(tmp, stem, pic, os.path.join(out, stem + '.pdf'))
        compile_pdf(tmp, name, combined(tex, pics), os.path.join(plots, name + '.pdf'))
    print('  panels -> %s, combined -> %s' % (out, os.path.join(plots, name + '.pdf')))


if __name__ == '__main__':
    main()
