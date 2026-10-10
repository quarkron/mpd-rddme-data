# Reversible reaction in a resolved sinusoidal field — manuscript figure

Builds the main-text figure for §"Reversible Bimolecular Reactions" of the LM-RDDME manuscript
(`manuscripts/LMRDDME_updated/LM-RDDME`), where it is `figures/pdf/rx_field.pdf`, label
`fig:rx-field-kl`. It replaces the former five-panel `rxl80_combo.pdf`, whose builder
(`../rxl80_combo_fig.py`) is kept for provenance and is no longer the figure of record.

## Build

```
docker run --rm -v /home/racda/ftsz_tests:/home/racda/ftsz_tests -v /raid/racda:/raid/racda \
  -w /home/racda/ftsz_tests/studies/lmrddme_validation/rx_field lm2.6:latest \
  bash -c 'export PATH=/raid/racda/opt/texlive/2026/bin/x86_64-linux:$PATH; \
           python3 rx_field_fig.py --install'
```

`--install` copies the result into the manuscript's `figures/pdf` and `figures/png`. TeX Live must
be on PATH or `usetex` falls back to mathtext and every font silently changes. There is no numpy or
matplotlib on the host — everything here runs in the container.

Four lettered blocks: **A** the setup cartoon, **B** the steady occupancy, **C** the divergence
from the reference in time, **D** the lattice itself. Only the blocks are lettered.

## Files

| file | what it is |
| --- | --- |
| `rx_field_fig.py` | assembles the three blocks and writes `out/fig_rx_field.{pdf,png}` |
| `rx_field_data.py` | every loader: profiles, counts, divergences, voxel fields |
| `rx_field_scheme.py` | block A, the setup cartoon |
| `rx_field_voxels.py` | block D, the lattice drawn as a solid of shaded voxel cells |
| `out/` | the rendered figure |

## The runs it reads (all already under `studies/`)

* `../mpd_crddme/rxl80c_out/{veryearly,early,late}_{excl,ideal}_s*.npz` — the lattice arms, from
  `../mpd_crddme/vex_reactive.py`. Three observation windows from the same initial condition at
  different `dt`: 30k steps to τ = 0.03, 60k to τ = 0.6, 150k to τ = 15; seeds 0 in the two fine
  windows and 0–3 in the production window.
* `../readdy_ref/out_field/rf_{vearly_,early_,}s*.npz` — the iPRD reference, 48 seeds.
* `../kl3way_cache.npz` — the divergence curves, from `../kl3way.py`.

Run parameters live only in each npz's `meta` string, not in any script: σ = 3/3/3.78 nm,
η = 0.088, K = 200 nm³, k_off = 5×10⁵ s⁻¹, γ = 0.3 kT, λ = 80 nm = 8h, 32³ voxels at h = 10 nm.

## Two things that were measured, not assumed

**The field axis is axis 0 of the reshaped occupancy.** `occA`/`occC` are the C-order ravel of the
HDF5 lattice with the particle axis summed away, and the solver's flat site index maps onto that
shape so the ramp lands on the *first* axis — the same convention `prof3[:, 0, :]` records.
Reshaping the other way averages over the ramp and returns a flat profile, which looks like a
result rather than a bug, so `rx_field_data.field_axis_check()` asserts it from the data on every
build.

**Block D has to average across the field.** A voxel holds ~3 particles, so its Poisson
fluctuation is 70 % of its mean while the field response is 17 %: any rendering of this lattice
shaded from raw per-voxel counts is visually pure noise. Probed at 1, 4, 8 and 32 voxels of depth
averaging before believing it. Block D therefore draws a real run of the lattice — two periods,
sixteen voxels, one stack deep — and shades each cube by the mean occupancy of the plane of voxels
normal to the field that it belongs to (1024 voxels, 2 % fluctuation). An earlier version drew the
whole 32³ box with voxels pooled 16×16 across the field; the stack says the same thing with far
less ink. `rx_field_voxels.draw_box` still takes any `(nx, ny, nz)` field, so the box version is
one argument away.

## Numbers the manuscript quotes from this figure

| quantity | iPRD | MPD-RDDME | point particles |
| --- | --- | --- | --- |
| profile peak-to-trough, A | 0.320 | 0.336 | 0.598 |
| profile peak-to-trough, C | 0.165 | 0.204 | 0.579 |
| late `N_A/N_A(0)` | 0.636 | 0.630 | 0.700 |
| late `N_C/N_A(0)` | 0.364 | 0.370 | 0.300 |

The profile amplitudes are what block B draws. The composition is quoted in the text but no longer
plotted, so `rx_field_fig.py` **prints the late-time counts on every build** — a number in the
manuscript should be reproducible by running the thing that made the figure.
