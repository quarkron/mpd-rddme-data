# Publication figure style

How every figure of the manuscript is drawn. The reference is Fig. 1 as finished in Illustrator
(`main/fig1/plots/figure1.pdf`). The settings below are implemented in `common/jcp_style.py`
(`PUB_RC`, `pub_page`, `si_page`, `pub_ax`, `pub_text`, `pub_header`, `pub_letter`, `pub_rule`, `pub_check`,
`pub_fonts`), so a script that uses them follows the
rules without restating them.

## 1. Type

- **Arial everywhere** (Helvetica is acceptable): text, numbers, axis labels, legends, and math. Never Computer
  Modern, DejaVu or a TeX font. Matplotlib runs with `text.usetex = False` and `mathtext.fontset = 'custom'` mapped
  to Arial, so `$\tau$` or `$k_{\mathrm{on}}$` is set in Arial too.
- **Symbols Arial lacks.** Angle brackets, over-bars (n with a bar), arrows and other math-only glyphs make matplotlib
  fall back to a TeX font. Write them in words ("occupancy / mean", "U minus its mean", "along x"); Greek letters,
  sub- and superscripts are fine. `jcp_style.pub_fonts(pdf)` lists the embedded fonts and flags anything not Arial.
- **Editable text.** Fonts are embedded as TrueType (`pdf.fonttype = 42`), so Illustrator opens the PDF with live
  ArialMT text and nothing has to be converted by hand.
- **Sizes**

  | element | size |
  |---|---|
  | panel letter (`a`, `b`, ...) | 15 pt, bold |
  | panel title | 13 pt, bold |
  | axis labels, in-panel headings | 11 pt |
  | tick labels, legends, annotations, color-bar labels | 10 pt |

  Nothing is smaller than 10 pt. The figure is printed smaller than drawn when it sits in the text, so anything
  below 10 pt becomes illegible.

## 2. Page

- Every combined main-text figure is an **8.5 x 11 in page** (SI figures: section 5). The content sits inside **0.5 in margins** (a 7.5 x 10 in box)
  and fills it: panels are sized to the page, not placed on it with white space around them.
- The combined file is `<folder>/plots/<name>.{pdf,png}`. Each panel is also written on its own, without letter or
  title, to `<folder>/plots/panels/<name>_<letter>.{pdf,png}` (`common/panel_export.py`).

## 3. Letters and titles

- The letter and its title are on **one line**: the bold letter at the panel's left edge, the bold title on the same
  baseline, centered over the panel (or immediately right of the letter when the panel is narrow). Never the letter on
  one line and the title under it.
- A title states the panel's message or subject in a few words. No numbers, symbols or units in titles.
- Side-by-side panels have their letters and titles on a common baseline; stacked panels share a left edge.
- Thin rules (0.5 pt, 30 % ink) may separate rows of panels.

## 4. Content

- **A panel carries data and short labels only.** Equations, parameter values and definitions (for example
  `sigma_C^3 = sigma_A^3 + sigma_B^3`, `K = 200 nm^3`, `D = 0.1 um^2/s`, `k_off = ...`) go in the caption, not on the
  figure. A schematic panel may name species and show sizes to scale, but its numbers belong to the caption.
- Use the space this frees for larger type and larger plots, and for a short in-panel heading that says what the
  simulation is.
- Every plot has a legend or direct labels; no frame around legends; legends sit in an empty corner or below.
- Physical names only: no file names, variable names, run tags or code identifiers anywhere a reader can see.
- Lines 1.5 pt, axes 0.8 pt, ticks inward. Keep the manuscript palette (`common/kip3_palette.py`).

- **PNG from the PDF when a label has a lone minus subscript.** Matplotlib's own PNG renderer drops a subscript that
  holds only a minus (k with a minus subscript comes out as a bare k); the PDF is correct. Fig. 5 therefore writes its
  PNGs by rasterizing the PDF with PyMuPDF (`fig_phase.png_from_pdf`); do the same in any figure with such a label.

## 5. Supplementary figures

The SI figures keep every rule of sections 1 and 4 (Arial, the sizes, no equations or parameter values in panels,
legends, physical names), with three differences:

- **No titles.** A panel carries its bold 15 pt letter only (`jcp_style.pub_letter`); its message is the caption's.
- **Any size.** An SI figure is as large as its content needs (`jcp_style.si_page(width, height)`), at most 7.5 in wide
  (the letter page's content box), with no forced 8.5 x 11 in page.
- Letters sit at the top-left corner of each panel, side-by-side panels on a common baseline.

`jcp_style.pub_check(fig, si=True)` checks the width limit and the 10 pt floor.

## 6. Files

Each figure folder writes `plots/<name>.pdf` (the combined figure), `plots/<name>.png` (300 dpi), one file per panel
in `plots/panels/`, and an Illustrator `.ai` beside every PDF (`tools/make_ai.py`, run by `render_all.sh`). The `.ai`
files open in Illustrator with live Arial text and editable paths; images inside the figure (kymographs, slice maps,
point clouds) stay embedded images.

## 7. Checks before installing

- `jcp_style.pub_check(fig)` (main) or `pub_check(fig, si=True)` (SI): the size limits and no text below 10 pt.
- `jcp_style.check_text_overlap(fig)`: no two labels overlapping.
- `jcp_style.pub_fonts(pdf)`: only ArialMT, Arial-BoldMT and Arial-ItalicMT embedded.

## 8. Fonts on a new machine

Arial is not shipped with Linux. Install Microsoft's core-fonts Arial (for example the `ttf-mscorefonts-installer`
package, or `arial32.exe` from the corefonts project extracted with `cabextract` into `~/.local/share/fonts`), then
clear matplotlib's font cache (`rm ~/.cache/matplotlib/fontlist-*.json`).
