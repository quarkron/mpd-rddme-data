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
