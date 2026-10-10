# main/fig1: `fig:schematic`

**Manuscript file:** figure1.pdf

**Published version:** the lattice schematic was printed from an HTML page by PDFKit (see the PDF metadata), and that
page was never saved. Only the published PDF/PNG are kept, in `plots/published/`.

**Proposed redraw (2026-10-05, not yet in the manuscript):** `scripts/make_figure1.py` generates the figure as TikZ in
the published figure's style, around the free-energy landscape U that every operator reads:

- a: lattice state evolution in a single timestep (diffuse -> structure -> react), U regenerated after every step
- b: reaction-diffusion for particles smaller than a subvolume
- c: mobility of hard spheres larger than a subvolume (fractional move; friction + U; dilute vs crowded)
- d: polymer relaxation (friction + U) and chemistry (U only), among crowders; FCC chain drawn on one checkerboard
  sublattice

**Re-render** (from `scripts/`, host venv with PyMuPDF; uses TeX Live 2026 at `/raid/racda/opt/texlive/2026`):

```bash
python make_figure1.py
```

**Outputs:** `plots/pdf/figure1.pdf` (full figure, 6.31 x 8.74 in, inside the 6.5 x 9 in text block) and
`plots/pdf/figure1_{a,b,c,d}.pdf` (each panel alone, without its letter and title); the same set at 300 dpi in
`plots/png/`. The generated `.tex` sources and LaTeX logs are in `build/`; edit the generator, never the `.tex`.

**Known gaps:** the U shading is illustrative: it is computed from the drawn content with invented weights
(one-body slope, sphere footprint, subunit, particle crowding), each panel stretched over its own range (c's sphere and
d's polymer on separate scales). It shows where U is high, not values.

**Producing the data:** none

**Data origin (copied into `data/`):** -
