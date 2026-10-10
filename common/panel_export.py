"""Export every lettered panel of a finished figure as a file of its own, without its letter or headline (2026-10-09).

    import panel_export
    fig.savefig(...)                                # the full figure first, as before
    panel_export.export(fig, 'polymer_r2')          # -> <figure folder>/plots/panels/polymer_r2_{a,b,...}.{pdf,png}

HOW A PANEL IS FOUND
    Letters: any text, figure-level or inside an axes, that reads as one letter and is bold or in parentheses --
    '\\textbf{b}', '(a)' -- set at 9 pt or more (a plain 'A' species label is not a letter). Every other piece of the figure (each axes with its ticks, labels and legend; each figure-level text,
    legend or patch) belongs to the nearest letter whose top is at or above its top edge and at or left of its left edge: the reading
    order of a lettered figure. A panel's file is the union of its pieces' boxes, cut out of the full canvas. An axes
    that reaches down past another block's letter (a colour bar shared by two blocks), and a figure-level key spanning
    two blocks of one row, go to <name>_shared.

WHAT IS LEFT OUT
    The letters themselves, the block headlines (jcp_style.block_note tags them gid='headline'), and the figure-level
    rules between blocks (Line2D artists on the figure). They are hidden for the export and restored after it, so the
    figure object is unchanged. A figure without letters is split into its side-by-side columns (<name>_left,
    <name>_right; the captions call them that); a single column is one panel and nothing is written.

WHERE
    plots/panels/ beside the script's own scripts/ folder (the figure folder of studies/vex-rddme), or `out_dir`.
"""
import os
import re
import sys

from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

LETTER = re.compile(r'^\(?([a-hA-H])\)?$')
PAD = 0.03                       # in, around each panel's box
TOL_Y, TOL_X = 0.16, 0.30        # in: a letter may sit this far below a piece's top / right of its left edge
SHARED = 0.5                     # in: an axes running this far past another block's letter is shared by both


def _letter_of(t):
    """A panel letter is BOLD ('\\textbf{a}' or fontweight bold) or in parentheses ('(a)'): a plain one-letter label
    inside a panel -- a species 'A', a motor 'M' -- is not one."""
    raw = t.get_text().strip()
    s = re.sub(r'\\textbf\{(.*)\}', r'\1', raw).strip()
    m = LETTER.match(s)
    if not (m and t.get_fontsize() >= 9 and t.get_visible()):
        return None
    w = t.get_fontweight()
    bold = '\\textbf' in raw or w == 'bold' or (isinstance(w, (int, float)) and w >= 600)
    return m.group(1).lower() if (bold or s.startswith('(')) else None


def _texts(fig):
    out = list(fig.texts)
    for ax in fig.axes:
        out += list(ax.texts)
    return out


def default_dir():
    here = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.normpath(os.path.join(here, '..', 'plots', 'panels'))


def export(fig, name, out_dir=None, dpi=300, exts=('pdf', 'png')):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inch = lambda bb: Bbox.from_extents(bb.x0 / fig.dpi, bb.y0 / fig.dpi, bb.x1 / fig.dpi, bb.y1 / fig.dpi)
    letters = []
    for t in _texts(fig):
        k = _letter_of(t)
        if k:
            letters.append((k, inch(t.get_window_extent(renderer=r)), t))
    print('  %s letters: %s' % (name, ' '.join('%s@(%.2f,%.2f)' % (k, L.x0, fig.get_figheight() - L.y1)
                                              for k, L, _ in letters)))
    if not letters:
        return _by_column(fig, name, r, inch, out_dir, dpi, exts)
    heads = [t for t in fig.texts if t.get_gid() == 'headline']
    rules = [a for a in list(fig.lines) + list(fig.artists) if isinstance(a, Line2D)]
    hide = [t for _, _, t in letters] + heads + rules

    def owner(bb):
        c = [(max(L.y1 - bb.y1, 0.0), -L.x0, k) for k, L, _ in letters      # the letter's TOP: a block's titles may
             if L.y1 >= bb.y1 - TOL_Y and L.x0 <= bb.x0 + TOL_X]             # sit level with or just above it
        if not c:                                        # nothing above-left: the nearest letter at all
            c = [((L.x0 - bb.x0) ** 2 + (L.y0 - bb.y1) ** 2, 0.0, k) for k, L, _ in letters]
        return min(c)[2]

    pieces = {}
    axleg = [ax.get_legend() for ax in fig.axes if ax.get_visible() and ax.get_legend() is not None]
    lay = [(g, g.get_in_layout()) for g in axleg]
    for g in axleg:                                      # an axes is measured without its legend; the legend is a
        g.set_in_layout(False)                           # piece of its own (it may sit over two blocks)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        bb = ax.get_tightbbox(r)
        if bb is None or bb.width <= 0:
            continue
        own = [k for k, _, t in letters if t.axes is ax]   # an axes that carries its own '(a)'
        b = inch(bb)
        k = own[0] if own else owner(b)
        # an axes reaching down past another block's letter (a colour bar two blocks share) belongs to neither:
        # it is written on its own as <name>_shared, so neither block's file runs into the other
        if not own and any(k2 != k and L.x0 <= b.x0 + TOL_X and b.y0 + SHARED < L.y1 < b.y1 - TOL_Y
                          for k2, L, _ in letters):
            k = 'shared'
        pieces.setdefault(k, []).append(b)
    for g, v in lay:
        g.set_in_layout(v)
    extra = [t for t in fig.texts if t not in hide and t.get_text().strip()] + list(fig.legends) + axleg
    extra += [a for a in fig.patches + fig.artists if a not in hide and not isinstance(a, Line2D)]
    for a in extra:
        try:
            bb = a.get_window_extent(renderer=r)
        except Exception:                                # an artist with no extent of its own
            continue
        if bb.width <= 0 or bb.height <= 0:
            continue
        b = inch(bb)
        k = owner(b)
        # a key or label crossing the boundary between two blocks of one row (a legend over both) is shared: it
        # covers the left edge of a block that has a neighbour to its left
        if any(b.x0 + TOL_X < L2.x0 < b.x1 - TOL_X and any(abs(L1.y1 - L2.y1) < 0.3 and L1.x0 < L2.x0 - TOL_X
                                                             for _, L1, _ in letters) for _, L2, _ in letters):
            k = 'shared'
        pieces.setdefault(k, []).append(b)

    out_dir = out_dir or default_dir()
    os.makedirs(out_dir, exist_ok=True)
    state = [(a, a.get_visible()) for a in hide]
    written = []
    try:
        for a in hide:
            a.set_visible(False)
        for k in sorted(pieces):
            u = Bbox.union(pieces[k])
            box = Bbox.from_extents(max(u.x0 - PAD, 0), max(u.y0 - PAD, 0),
                                    min(u.x1 + PAD, fig.get_figwidth()), min(u.y1 + PAD, fig.get_figheight()))
            for e in exts:
                q = os.path.join(out_dir, '%s_%s.%s' % (name, k, e))
                fig.savefig(q, dpi=dpi, bbox_inches=box)
                written.append(q)
            print('  %s panel %s: %.2f x %.2f in' % (name, k, box.width, box.height))
    finally:
        for a, v in state:
            a.set_visible(v)
    print('  panels -> %s' % out_dir)
    return written


def _by_column(fig, name, r, inch, out_dir, dpi, exts):
    """No letters: panels side by side that the caption calls left and right. Axes whose horizontal extents overlap
    (twins, insets) are one panel; figure-level texts and legends go to the column holding their centre. One column
    is one panel, and nothing is written."""
    cols = []
    for ax in sorted((a for a in fig.axes if a.get_visible()), key=lambda a: a.get_position().x0):
        p = ax.get_position()
        for c in cols:
            if p.x0 < c['x1'] and c['x0'] < p.x1:
                c['ax'].append(ax); c['x0'] = min(c['x0'], p.x0); c['x1'] = max(c['x1'], p.x1); break
        else:
            cols.append(dict(x0=p.x0, x1=p.x1, ax=[ax]))
    if len(cols) < 2:
        print('  %s: one panel, nothing exported' % name)
        return []
    keys = ['left', 'right'] if len(cols) == 2 else [str(i + 1) for i in range(len(cols))]
    boxes = [[inch(a.get_tightbbox(r)) for a in c['ax']] for c in cols]
    W = fig.get_figwidth()
    heads = [t for t in fig.texts if t.get_gid() == 'headline']
    for a in [t for t in fig.texts if t.get_text().strip() and t not in heads] + list(fig.legends):
        b = inch(a.get_window_extent(renderer=r))
        xc = 0.5 * (b.x0 + b.x1) / W
        j = min(range(len(cols)), key=lambda i: 0 if cols[i]['x0'] <= xc <= cols[i]['x1']
                else min(abs(xc - cols[i]['x0']), abs(xc - cols[i]['x1'])))
        boxes[j].append(b)
    out_dir = out_dir or default_dir()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    state = [(t, t.get_visible()) for t in heads]
    for t in heads:
        t.set_visible(False)
    for k, bb in zip(keys, boxes):
        u = Bbox.union(bb)
        box = Bbox.from_extents(max(u.x0 - PAD, 0), max(u.y0 - PAD, 0), min(u.x1 + PAD, W),
                                min(u.y1 + PAD, fig.get_figheight()))
        for e in exts:
            q = os.path.join(out_dir, '%s_%s.%s' % (name, k, e))
            fig.savefig(q, dpi=dpi, bbox_inches=box)
            written.append(q)
        print('  %s panel %s: %.2f x %.2f in' % (name, k, box.width, box.height))
    for t, v in state:
        t.set_visible(v)
    print('  panels -> %s' % out_dir)
    return written
