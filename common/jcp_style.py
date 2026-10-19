"""Figure geometry and type sizes for AIP / Journal of Chemical Physics.

Figures are built at their FINAL printed size rather than drawn large and scaled down in LaTeX.
Scaling a figure scales its type with it: an 11 in wide canvas placed in a 6.69 in column shrinks
every label by 0.6, which pushes a 9.5 pt legend to 5.7 pt and under the 8 pt minimum AIP sets for
labels and legends. Drawing at final size means \\includegraphics applies no scale factor at all,
so what is set here is what appears on the page.

AIP limits: 3.37 in single column, 6.69 in double column, 8.25 in maximum depth, 8 pt minimum type.
"""
import matplotlib

COL = 3.37          # single-column width, in
FULL = 6.69         # double-column width, in
MAXDEPTH = 8.25     # maximum figure depth, in

# Every size here is >= 8 pt so that no label falls below the AIP floor at final size.
RC = {
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.45,
    'ytick.minor.width': 0.45,
    'lines.linewidth': 1.1,
    'lines.markersize': 3.6,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.6',
    'legend.handlelength': 1.7,
    'legend.columnspacing': 1.0,
    'legend.handletextpad': 0.5,
    'legend.borderpad': 0.35,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    'figure.dpi': 300,
}


# Categorical series colors, checked rather than chosen by eye. The default matplotlib cycle puts
# green next to red, whose separation under deuteranopia is a color difference of 3.9 on the scale
# where 8 is the target and 6 is the floor, so those two series are indistinguishable to a reader
# with the most common form of color blindness. This is an Okabe-Ito subset, which passes the
# lightness band, chroma floor, colorblind separation, normal-vision separation and surface contrast
# checks at four series. Assign these in fixed order and never cycle past the end: a fifth series
# means faceting or folding, not a generated hue.
CATEGORICAL = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']

# Reference lines and exact results are drawn in ink, not in a series color, so that a measurement
# and the thing it is measured against are never confused by hue alone.
REFERENCE = '0.25'
MUTED = '0.55'


def apply(categorical=False):
    """Apply the house geometry and type sizes.

    The series palette is OPT IN. Existing figures in this manuscript select colors through the
    default cycle by name, so changing that cycle here would silently restyle every figure the next
    time it is regenerated, including ones already placed in the document. New figures pass
    categorical=True, or read CATEGORICAL directly; existing ones keep what they had until someone
    decides to change them on purpose.
    """
    matplotlib.rcParams.update(RC)
    if categorical:
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=CATEGORICAL)


# ---------------------------------------------------------------------------------------------
# Sans-serif variant, and the one-line takeaway each lettered block carries.
# ---------------------------------------------------------------------------------------------

# ★Under usetex the family switch has to happen in TeX. matplotlib wraps the label in \sffamily
# when font.family is sans-serif, which moves the TEXT; MATH keeps Computer Modern Roman unless
# sansmath is loaded and switched on, and the result is a figure whose words are sans and whose
# every $n$, $\tau$ and $\eta$ is serif. Loading it here is what keeps the two together.
SANS_PREAMBLE = r'\usepackage{sansmath}\sansmath'

# Computer Modern Sans is the sans companion of the Computer Modern the manuscript body is set in,
# so a sans figure still reads as belonging to the same document. DejaVu Sans is the fallback the
# no-latex path lands on.
SANS_FAMILY = ['Computer Modern Sans serif', 'DejaVu Sans']

NOTE_SIZE = 8.0          # pt, the block takeaway: the AIP floor, since it is a label like any other
NOTE_COLOR = '0.15'      # near-black, so the sentence is subordinate to the block letter beside it


def sans(rc, tex):
    """Return `rc` turned sans-serif, text and math together.

    Opt-in per figure rather than applied in RC: the figures already placed in the manuscript were
    drawn serif, and flipping the shared default would restyle every one of them the next time it
    was regenerated -- the same reason the categorical palette in apply() is opt-in.
    """
    rc = dict(rc)
    rc['font.family'] = 'sans-serif'
    rc['font.sans-serif'] = list(SANS_FAMILY)
    if tex:
        pre = rc.get('text.latex.preamble', '') or ''
        if SANS_PREAMBLE not in pre:
            rc['text.latex.preamble'] = (pre + '\n' + SANS_PREAMBLE).strip()
    else:
        rc['mathtext.fontset'] = 'dejavusans'      # or the fallback math is serif again
    return rc


def block_note(fig, x, y, text, size=NOTE_SIZE, color=NOTE_COLOR, bold=True, **kw):
    """The one-line conclusion that sits beside a block letter, at the letter's own baseline.

    It goes on the LETTER'S ROW rather than under the block: a caption line beneath each block
    would cost about 0.2 in five times over, and these figures are already drawn against the AIP
    depth limit with a caption that has to fit on the same page. The letter row is the one band
    that is already reserved and mostly empty.

    ★Bold has to be set in the TeX. Under usetex matplotlib ignores fontweight entirely -- the same
    trap the block letters carry their own \textbf for -- so the sentence is wrapped here and the
    rcParam is left for the no-latex path. A note is therefore PLAIN TEXT: math inside \textbf
    stays medium weight and the line would come out half bold, which is why these sentences carry
    no symbols and no numbers (the caption is where a number belongs).
    """
    # A '\n' is drawn as separate texts, one per line, rather than handed to matplotlib as
    # multi-line: under usetex each line is its own \textbf, and the leading is then a number
    # here rather than whatever the TeX box decides.
    out = []
    for i, line in enumerate(str(text).split('\n')):
        if bold:
            line = (r'\textbf{%s}' % line) if matplotlib.rcParams['text.usetex'] else line
            kw.setdefault('fontweight', 'bold')
        out.append(fig.text(x, y - i * 1.25 * size / 72.0 / fig.get_figheight(), line,
                            fontsize=size, color=color, ha='left', va='top', **kw))
        out[-1].set_gid('headline')                     # panel_export leaves these out of the per-panel files
    return out[0] if len(out) == 1 else out


def check_width(fig, textw, label='figure'):
    """Report the drawn width against the text width, and name what sticks out.

    ★These figures are saved with savefig.bbox = 'tight', so the canvas width is a REQUEST and the
    file's width is whatever the artists span. A block note that runs past the last panel therefore
    widens the file rather than being clipped, \\includegraphics scales nothing, and the graphic
    overflows the text block by however much the sentence overhangs -- silently, since LaTeX only
    warns about overfull boxes it builds itself. Measured every run, like the block gaps."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    w = bb.x1 - bb.x0
    flag = '  << OVER the %.2f in text width' % textw if w > textw + 0.005 else ''
    print('  %s width %.3f in%s' % (label, w, flag))
    if flag:
        for t in fig.texts:
            e = t.get_window_extent(renderer=r)
            x1 = e.x1 / fig.dpi - bb.x0
            if x1 > textw:
                print('    overhang %.3f in: %r' % (x1 - textw, t.get_text()[:60]))
    return w


def check_text_overlap(fig, label='figure'):
    """Report any two figure-level texts whose boxes intersect.

    A block note is set in a margin the layout never measured, so the failure mode is a sentence
    running under the NEXT block's letter -- which renders perfectly and reads as nonsense. Cheap
    enough to run on every build; only figure-level texts and axis labels are checked, since that is
    where the notes live."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    arts = list(fig.texts)
    for a in fig.axes:
        arts += [a.xaxis.label, a.yaxis.label, a.title]
    boxes = [(t.get_text(), t.get_window_extent(renderer=r))
             for t in arts if t.get_text().strip()]
    hits = 0
    for m in range(len(boxes)):
        for n in range(m + 1, len(boxes)):
            (ta, ba), (tb, bb) = boxes[m], boxes[n]
            if ba.x0 < bb.x1 and bb.x0 < ba.x1 and ba.y0 < bb.y1 and bb.y0 < ba.y1:
                print('  %s TEXT OVERLAP: %r and %r' % (label, ta[:52], tb[:52]))
                hits += 1
    return hits


# ---------------------------------------------------------------------------------------------
# Publication layout (2026-10-19, FIGURE_STYLE.md): Arial throughout, 10-11 pt text, a bold 13 pt
# panel title on the same line as its bold 15 pt letter, every figure on an 8.5 x 11 in page.
# This supersedes the AIP/LaTeX settings above for the figures restyled to it.
# ---------------------------------------------------------------------------------------------
PAGE_W, PAGE_H = 8.5, 11.0       # in, the letter page every combined figure is drawn on
MARGIN = 0.5                     # in, on every side; the content box is 7.5 x 10 in
TEXT_PT = 10                     # tick labels, legends, annotations
LABEL_PT = 11                    # axis labels and in-figure titles
TITLE_PT = 13                    # panel title, bold
LETTER_PT = 15                   # panel letter, bold
INK = '#1b2430'

PUB_RC = {
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'],
    'mathtext.fontset': 'custom',             # math in Arial too: no Computer Modern anywhere
    'mathtext.rm': 'Arial',
    'mathtext.sf': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'mathtext.default': 'it',
    'pdf.fonttype': 42,                       # TrueType, so Illustrator sees editable ArialMT text
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': TEXT_PT,
    'axes.labelsize': LABEL_PT,
    'axes.titlesize': LABEL_PT,
    'xtick.labelsize': TEXT_PT,
    'ytick.labelsize': TEXT_PT,
    'legend.fontsize': TEXT_PT,
    'figure.titlesize': TITLE_PT,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'legend.frameon': False,
    'legend.handlelength': 1.8,
    'legend.borderpad': 0.3,
    'legend.labelspacing': 0.3,
    'axes.edgecolor': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'text.color': INK,
    'savefig.bbox': None,                     # keep the full page
    'figure.dpi': 300,
}


def pub_page():
    """A new 8.5 x 11 in figure for a main-text figure (call inside plt.rc_context(PUB_RC))."""
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(PAGE_W, PAGE_H))


def si_page(width, height):
    """A new SI figure of the size its content needs (at most SI_W wide; call inside plt.rc_context(PUB_RC))."""
    import matplotlib.pyplot as plt
    return plt.figure(figsize=(width, height))


SI_W = 7.5                       # in, the widest an SI figure may be (the content box of the letter page)


def pub_ax(fig, x, ytop, w, h):
    """An axes placed in INCHES from the figure's top-left corner."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    return fig.add_axes([x / W, 1.0 - (ytop + h) / H, w / W, h / H])


def pub_text(fig, x, ytop, text, **kw):
    """Figure text placed in inches from the figure's top-left corner."""
    return fig.text(x / fig.get_figwidth(), 1.0 - ytop / fig.get_figheight(), text, **kw)


def pub_header(fig, letter, title, x, ytop, w, center=True):
    """Panel letter (bold 15 pt) at x and its title (bold 13 pt) on the same baseline, the title centred over the
    panel's width w (from x) or, with center=False, starting right after the letter. ytop = the line's top, in.
    SI figures pass title=None: the letter alone."""
    W, H = fig.get_figwidth(), fig.get_figheight()
    y = 1.0 - (ytop + LETTER_PT / 72.0) / H
    lt = fig.text(x / W, y, letter, fontsize=LETTER_PT, fontweight='bold', ha='left', va='baseline', color=INK)
    tt = None
    if title:
        if center:
            tt = fig.text((x + 0.5 * w) / W, y, title, fontsize=TITLE_PT, fontweight='bold', ha='center',
                          va='baseline', color=INK)
        else:
            tt = fig.text((x + 0.32) / W, y, title, fontsize=TITLE_PT, fontweight='bold', ha='left',
                          va='baseline', color=INK)
        tt.set_gid('headline')                 # panel_export leaves titles out of the per-panel files
    return lt, tt


def pub_letter(fig, letter, x, ytop):
    """An SI panel letter (bold 15 pt) with its top at ytop, in inches from the figure's top-left; no title."""
    return pub_header(fig, letter, None, x, ytop, 0.0)[0]


def pub_rule(fig, ytop, x0=MARGIN, x1=None):
    """A thin separator between panel rows."""
    from matplotlib.lines import Line2D
    W, H = fig.get_figwidth(), fig.get_figheight()
    x1 = W - MARGIN if x1 is None else x1
    fig.add_artist(Line2D([x0 / W, x1 / W], [1.0 - ytop / H] * 2, transform=fig.transFigure, color=INK, lw=0.5,
                          alpha=0.3))


def pub_check(fig, label='figure', si=False):
    """Main text: everything inside the page margins. SI (si=True): the content at most SI_W wide. Both: no text
    below TEXT_PT (bold titles and letters excepted)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = fig.get_tightbbox(r)
    W, H = fig.get_figwidth(), fig.get_figheight()
    if si:
        ok = (bb.x1 - bb.x0) <= SI_W + 0.02
        where = '  << WIDER than %.1f in' % SI_W
    else:
        ok = bb.x0 >= MARGIN - 0.02 and bb.x1 <= W - MARGIN + 0.02 and bb.y0 >= MARGIN - 0.02 \
            and bb.y1 <= H - MARGIN + 0.02
        where = '  << OUTSIDE the %.1f in margins' % MARGIN
    small = sorted({round(t.get_fontsize(), 1) for t in fig.findobj(matplotlib.text.Text)
                    if t.get_text().strip() and t.get_visible() and t.get_fontsize() < TEXT_PT - 0.01})
    print('  %s content %.2f-%.2f x %.2f-%.2f in on the %.2f x %.2f canvas%s%s'
          % (label, bb.x0, bb.x1, H - bb.y1, H - bb.y0, W, H, '' if ok else where,
             ('  << type below %d pt: %s' % (TEXT_PT, small)) if small else ''))
    return ok and not small


def pub_fonts(pdf_path, label=None):
    """After saving: list the fonts the PDF embeds and flag anything that is not Arial (mathtext falls back to
    Computer Modern for glyphs Arial lacks, such as angle brackets, over-bars and arrows; write those in words)."""
    import fitz
    names = sorted({f[3].split('+')[-1] for f in fitz.open(pdf_path)[0].get_fonts()})
    bad = [n for n in names if not n.startswith('Arial')]
    print('  %s fonts: %s%s' % (label or pdf_path, ', '.join(names), ('  << NOT ARIAL: %s' % bad) if bad else ''))
    return not bad
