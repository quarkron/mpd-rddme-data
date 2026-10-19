# TCR-pMHC transport in an immunological synapse

**Moved 2026-09-10** from `studies/lmrddme_validation/synapse` to `studies/tcr-pmhc`; the
campaign tree moved with it, `/raid/racda/campaigns/lmrddme_validation/synapse` ->
`/raid/racda/campaigns/tcr_pmhc`. ★The wrappers derive the repo root by relative depth, and
the new directory is one level shallower, so `../../..` became `../..`; without that fix the
container mounts `/home/racda` and no script is found. `tcr_pmhc.ipynb` drives the study and
reimplements none of it.

Rebuild of Isaacson & Zhang, *Bull. Math. Biol.* **87**:14 (2024), section IV.D, with hard-sphere
volume exclusion. Their CRDDME has no two-body interactions; that section says in as many words
that they therefore "ignore size-based segregation forces", and it attributes the gap between
their seconds-scale core filling and the physiological tens of minutes to the larger size of the
complexes and their repulsion by other synapse components. This study tests that.

Everything lives in `studies/lmrddme_validation/synapse/`. Nothing was written into
`Lattice_Microbes_2.6/` except two regenerated `__pycache__` byte-code files in a root-owned
directory, from the first import attempt before the private jLM existed.

**Binary provenance, and a hazard.** Another session was editing the LM tree and rebuilding
`build_container` while this study ran: `StructureFreeVolume.cu` and `struct_freevol_dev.cuh`
(the note-5 free-volume functional) were being added, and `_lm...so` was relinked at 15:39
local, between the first rate gate and the 200 s production runs. The production runs all
started at 15:44 and finished by 15:53, after that relink and before any later one, so they used
one binary throughout. The rate gate was re-run against the current binary afterwards and passes
at 1.002. Anyone repeating this should re-run `synapse_rate_gate.py` first, because the build
underneath is moving.

## THE CANONICAL RUN

Settled by owner direction 2026-09-07. Everything else in this file is either a gate, a control,
or an exploratory arm that was run and set aside; this is the model.

- **Gauge `db`.** Free TCR and pMHC experience no active force, only the complex couples to the
  ramp, exactly as Isaacson & Zhang specify.
- **f0 = 1 kT/um**, their stated value read in kT per micron. Dharan & Farago's 0.1 pN, which is
  24.1 kT/um, was run and set aside; see the section on it below for why.
- **Density 1410 per um^2** of free TCR and pMHC seeded outside the core, their initialization.
- **40 nm voxels, 288 x 288 x 8**, one active z-slice, disk of radius 5.6 um, core 2.0 um,
  pSMAC edge 4.0 um, 200 s, adaptive tau off.
- Three species: free TCR, free pMHC, TCR-pMHC. LFA-1-ICAM-1 and CD45 are implemented and unused.

Two commands reproduce it, and a third and fourth give the transport-only control that carries
the exact Boltzmann reference:

```
./lmrun_bg.sh 0 L200_db_point    python synapse_run.py --arm point    --gauge db --tsim 200 --frames 60 --density 1410 --out out/L200_db_point.npz
./lmrun_bg.sh 1 L200_db_excluded python synapse_run.py --arm excluded --gauge db --tsim 200 --frames 60 --density 1410 --out out/L200_db_excluded.npz
./lmrun_bg.sh 2 T200_db_point    python synapse_run.py --arm point    --gauge db --no-reaction --preseed-complex 1410 --tsim 200 --frames 60 --out out/T200_db_point.npz
./lmrun_bg.sh 3 T200_db_excluded python synapse_run.py --arm excluded --gauge db --no-reaction --preseed-complex 1410 --tsim 200 --frames 60 --out out/T200_db_excluded.npz
```

Run `synapse_rate_gate.py` first if the LM build has moved underneath. Figures:
`fig_T200_db.pdf` is the result, `fig_L200_db.pdf` is the reactive version of it.


## Files

| file | role |
|---|---|
| `synapse_model.py` | geometry, species table, equivalent diameters, gauge, rates. Runnable: prints the table. |
| `synapse_run.py` | builds and runs one arm, reduces frames to radial histograms |
| `synapse_rate_gate.py` | absolute gate of the bimolecular rate against well-mixed kinetics |
| `synapse_analyze.py` | inspection and the comparison figure |
| `lmrun.sh` | container wrapper (mounts `/lm`, `/ftsz`, `/raid/racda/campaigns`) |
| `build_jlm.sh` | rebuilds the private jLM; needed once per container change |
| `out/` | symlink to `/raid/racda/campaigns/lmrddme_validation/synapse` |

## Configuration

40 nm voxels, 288 x 288 x 8 sites, disk of radius 5.6 um cut by a site mask, core at 2.0 um and
pSMAC edge at 4.0 um. Free TCR and pMHC at 0.10 um^2/s, complex at 0.06, reversible binding at
k_on 2e4 /M/s and k_off 0.1 /s, seeded at 1410 per um^2 outside the core, which is their
initialization. dt = 1.28e-4 s from q_stab 0.008.

## Six things that were measured, not assumed

**1. A one-voxel layer is impossible; use eight slices with one active.** The MPD kernel rejects
a z dimension not divisible by 8 (and an x dimension not divisible by 32). The layer is made one
voxel thick by sealing the other seven slices into the inert `out` region, which costs empty
voxels and nothing else.

**2. jLM reverses its axes on the way to the solver.** `RDME.resizeLattice` does
`nx,ny,nz = dimensions` then `ByteLattice(nz,ny,nx,...)`, so the LAST entry of `dimensions` is
the axis the kernel calls x. Masks and `placeNumber` follow the same reversed convention. This
module holds everything in solver order and converts at the boundary with `to_jlm`. For a cube
the error is silent, which is why the microtubule notebook this was built from brute-forces the
permutation at analysis time instead of knowing it.

**3. `sim.reaction(..., regions=None)` silently disables the reaction.** The default is not "all
regions", it is "no regions": the location matrix stays empty, the reaction never fires, and the
run completes, conserves mass, and produces a perfectly plausible diffusion-only trajectory with
exactly zero product. Asserted in `build` rather than trusted.

**4. Adaptive tau makes the chemistry about 1/tau_scale times too fast per unit reported clock.**
With `setAdaptiveTau(1, 0.2, 0.05)` the solver drives the scale to its 0.05 floor and the
well-mixed gate then reads 18x the correct product at early times, decaying as it saturates. With
adaptive tau off the same gate reads 0.99. It is off by default here and must stay off. The bare
timestep is already safe: hop probability 0.008 and reaction probability 3e-4 per step. This is
the same rates-versus-clock failure recorded for the FtsZ campaign, in the opposite direction.

**5. The bimolecular rate is right, to 0.1%, once 3 and 4 are fixed.** `synapse_rate_gate.py`
seeds the whole disk uniformly with the field off and compares the product count against the
integrated well-mixed kinetics. Production geometry and density: median ratio 0.999.

**6. A one-body potential cannot act on transport alone.** Every acceptance in this scheme takes
Delta U, reactions included, so coupling only the complex to the ramp makes the binding
equilibrium position dependent, K(r) = K_id exp(-beta Phi_C(r)). Phi_C spans 7.2 kT across their
domain, a factor of 1340 in K. Measured: the field-free gate reproduces the analytic kinetics to
1%, and the identical run at f0 = 1 kT/um reproduces 6.9% of it, against 6.0% predicted by the
area-weighted mean of exp(-psi). There is no switch for this and there should not be. It applies
to any scheme claiming detailed balance of reaction fluxes under a one-body potential, CRDDME
included, so it is a finding about the method class and not about this implementation.

**The model is the `db` gauge**, by owner direction 2026-09-07: free TCR and pMHC experience no
active force, exactly as Isaacson & Zhang specify ("we assume that free TCRs and pMHCs do not
experience any active forces, and simply diffuse"), and only the complex couples to the ramp.

That is not a limitation of this rebuild, it is a property of their published model. They state
that their scheme preserves detailed balance of reaction fluxes, and they put the active force on
the complex alone. Those two commitments together force K(r) = K_id exp(-beta Phi_C(r)). The
effect is theirs, and their 20 s window is far too short to reveal it.

`neutral`, which gives each free species half the complex's coupling so the reaction's Delta U
vanishes identically, is retained ONLY as the diagnostic that isolates the effect. It is not the
model, because it would give the free species a drift their model gives them none of. Its rate
gate reads 0.992 against 0.069 for `db`, which is the measurement of the effect.

## The diameter handed to the functional

Two steps, both parameter choices, neither touching a functional.

1. Each ectodomain is a spherocylinder, not a sphere. The Kihara convex-body excluded volume
   gives the equivalent sphere: 5.3 nm for free TCR and pMHC, 8.8 nm for the complex, 9.9 nm for
   CD45, 16.0 nm for LFA-1-ICAM-1. These agree with the Tirado-Garcia de la Torre Stokes
   equivalents to better than 7% for the three complexes.
2. The molecules are anchored, so their crowding is areal, but the functional reads a volume
   fraction out of a cubic voxel and would spread a jammed monolayer through 40 nm of z. Matching
   the second virial coefficient per unit area of the slab gives
   `sigma_eff = (0.75 h sigma^2)^(1/3)`: 9.5, 13.3, 14.3 and 19.7 nm. The jamming-limit
   convention replaces 0.75 by 1.5, raising every diameter by 2^(1/3) and leaving every size
   ratio unchanged. `--packing-match` selects it.

The size ratio between the two complexes is 1.49 either way, inside the window where the paper's
two-species figure is quantitative rather than overdoing the asymmetry.

## Results, 2026-09-07

All at 40 nm voxels, 288 x 288 x 8, their disk and their initialization of 1410 per um^2, 200 s,
`db` gauge. Mass conserved exactly in every run, zero word-bound violations everywhere.

### Transport only, against an exact reference (`T200_db_*`)

Complex pre-seeded uniformly at 1410 per um^2, no reaction, so the stationary profile is exactly
Boltzmann in the ramp. This validates the drift machinery in the production geometry with nothing
fitted, and it is the cleanest statement of what exclusion does.

| | area-weighted departure from Boltzmann | central density (um^-2) | central areal packing |
|---|---|---|---|
| Boltzmann, exact | - | 22323 | **1.37** |
| point particles | **0.012** | 20859 | **1.28** |
| volume excluded | 0.475 | 6406 | 0.39 |

The point arm lies on the analytic curve to 1.2%, flat in the ratio across the whole domain. That
is the drift kernel validated against an exact answer at production scale.

**The headline.** The equilibrium the ramp alone demands is geometrically impossible: an areal
packing fraction of 1.37 at the centre, against 0.82 for random close packing of disks. The
point-particle description is not inaccurate there, it is unphysical, and it reaches 1.28 because
nothing stops it. Volume exclusion caps the centre at 0.39 and pushes the displaced material
outward, suppressing the central density 3.5-fold and raising the rim 2.9-fold. This is the
qualitative-not-a-correction result the application section needs. Figure: `fig_T200_db.pdf`.

### Reactive, their full model (`L200_db_*`)

| | complexes in core | total complexes | core areal packing | max site occupancy |
|---|---|---|---|---|
| point particles | 22571 | 36092 | 0.148 | 25 |
| volume excluded | 18122 | 33139 | 0.122 | 17 |

Both arms plateau by about 150 s, so 200 s is an adequate window and no longer run is needed.
Exclusion cuts core occupancy by 20% and flattens the central density from 5300 to 2900 per um^2,
a factor of 1.8. Figure: `fig_L200_db.pdf`.

**A negative result worth keeping.** Exclusion changes the PLATEAU, not the approach: both arms
rise together to about 50 s and have the same half-time. So at this density, on monomeric
complexes of a single size, exclusion does not convert their seconds into the physiological tens
of minutes. It caps how full the core gets, it does not slow the filling. Their hypothesis was
about MICROCLUSTERS, which are larger objects than anything here, so this tests only the part of
it that a single-size model can test.

### The exclusion ceiling, across both gauges

Central complex density in the innermost radial bin at 200 s, and the peak site occupancy
reached anywhere in the run, over all four reactive arms. Areal fraction uses the true
equivalent-sphere diameter of 8.83 nm, so it is the physical coverage of the membrane.

| run | central density (um^-2) | central areal fraction | peak site occupancy |
|---|---|---|---|
| neutral, point | 14531 | **0.889** | **46** |
| neutral, excluded | 5039 | 0.308 | 17 |
| db, point | 5352 | 0.328 | 25 |
| db, excluded | 2969 | 0.182 | 17 |

The `neutral` gauge removes the radius-dependent suppression of binding, so the reaction runs
nearly to completion (95860 complexes against 94074, statistically the same in both arms) and the
whole population is available to pile into the core. The point arm then reaches an areal fraction
of 0.889 at the centre, above random close packing of disks at 0.82 and approaching the hexagonal
limit of 0.907, which no fluid can do. Both excluded arms cap at a peak site occupancy of 17
regardless of gauge, against 25 and 46 for the corresponding point arms. That ceiling, identical
across two very different chemistries, is the cleanest single signature of the functional doing
its job.

### Replicated, three seeds each (2026-09-07)

Every number above was single-seed. Seeds 0, 1, 2 of the canonical configuration, quoted as mean
and sample standard deviation. All eight replicate runs finished with zero word-bound violations.

**Transport only**, complexes pre-seeded at 1410 per um^2, no reaction, so the stationary state is
exactly Boltzmann in the ramp.

| | complexes in core | core areal packing | central density (um^-2) | peak site occupancy |
|---|---|---|---|---|
| point particles | 87004 +- 366 | 0.4236 +- 0.0018 | 21510 +- 733 | 54.7 +- 0.6 |
| volume excluded | 54409 +- 305 | 0.2649 +- 0.0015 | 6328 +- 78 | 15.0 +- 0.0 |

Central areal packing fraction: the exact Boltzmann profile demands **1.37**, the point arm
reaches **1.32 +- 0.04**, the excluded arm sits at **0.388 +- 0.005**. Random close packing of
disks is 0.82. The point arm agrees with the analytic curve to 3.6% at the centre and to 1.2%
area-weighted, so it is right about an equilibrium that cannot exist.

The peak site occupancy is the cleanest single number in the study: 54.7 +- 0.6 of the 64
available slots without exclusion, and exactly 15.0 +- 0.0 with it, identical in all three seeds.
At the Dharan & Farago force the same point-particle arm passes 64 and the run dies.

**Reactive**, their full model.

| | complexes in core | complexes total | core areal packing | central density (um^-2) | peak occupancy |
|---|---|---|---|---|---|
| point particles | 22685 +- 99 | 36248 +- 146 | 0.1480 +- 0.0002 | 5247 +- 363 | 25.0 +- 2.0 |
| volume excluded | 18215 +- 232 | 33188 +- 143 | 0.1220 +- 0.0011 | 2930 +- 39 | 16.7 +- 1.5 |

Exclusion suppresses the central density by 3.40x in transport and 1.79x with the reaction, and
cuts core occupancy by 37% and 20% respectively. Seed-to-seed spread is 1 to 3% throughout, so
every one of these separations is far outside it.


### Spatial profiles, the two-row version of their Figure 8 (2026-09-07)

`synapse_maps_fig.py` builds the direct analogue of Isaacson & Zhang Fig. 8, complex profiles
across the contact zone at a sequence of times, with point particles on the top row and volume
exclusion on the bottom, sharing one colour axis. Runs `M_react_*` and `M_trans_*`, canonical
configuration, maps stored at t = 0, 2, 10, 30, 100, 200 s.

Two presentation decisions, both load-bearing:

- **Plotted as surface density, not counts.** Their colour axis is molecules per mesh polygon of
  7093 nm^2; our voxel is 1600 nm^2, so raw counts differ by 4.4x for identical physics. Their
  0-20 per polygon is 0-2820 per um^2.
- **Displayed at their cell size.** A 40 nm voxel holds about two complexes, so voxel-resolved
  counts are 70% shot noise and the panel is pure speckle. Block-summing 2x2 gives 80 nm cells of
  6400 nm^2, within 10% of their polygon, which is the honest way to put the figures side by
  side. `--coarsen 1` shows the raw voxel field.

Peak complex surface density in um^-2, transport-only arm:

| t (s) | point particles | volume excluded | ratio |
|---|---|---|---|
| 0 | 3594 | 3594 | 1.00 |
| 10 | 6719 | 4219 | 1.59 |
| 30 | 14531 | 6406 | 2.27 |
| 100 | 22344 | 7031 | 3.18 |
| 200 | 25000 | 7031 | **3.56** |

At 200 s the point arm peaks at an areal packing fraction of **1.53** and the excluded arm at
**0.43**. The panels carry a white contour at 13397 per um^2, the density at which these
molecules reach random close packing of disks. It appears in the point-particle row at 100 and
200 s and nowhere in the excluded row: inside that contour the point-particle description is
reporting a state denser than a monolayer of these molecules can physically be.

The reactive arm shows the same effect more weakly, peak ratio 1.39 at 200 s and peak packing
0.37 against 0.27, because the db gauge suppresses binding toward the centre and the complexes
never reach the densities the transport-only arm does. Figures: `fig_maps_react.pdf` is the deliverable, owner direction 2026-09-07;
`fig_maps_trans.pdf` is the same construction for the transport-only arm.

**Typesetting.** Set in Computer Modern through LaTeX, matching the manuscript body: `build/main.pdf`
embeds CMR12/CMR8/CMBX12/CMSY10, so a figure in DejaVu Serif would not match the page. The script
uses the same rc block every other figure in this paper uses, jcp_style.RC plus inward ticks on all
four sides plus `text.usetex` with Computer Modern Roman, and falls back to `mathtext.fontset: cm`
only if latex is missing. It PRINTS which of the two it used, because the fallback is otherwise
silent and changes every glyph.

★TeX Live is at `/raid/racda/opt/texlive/2026` on the host and was NOT visible inside the
container, so usetex fell back without saying so. `lmrun.sh` and `lmrun_bg.sh` now mount
`/raid/racda/opt` read-only, put the TeX Live bin directory on PATH and set `TMPDIR=/raid/racda/tmp`
(the root filesystem is full). latex and dvipng resolve; ghostscript does not, which matters only
for PS output.

**Text placement.** Three collisions were fixed by measuring rather than guessing.
The shared x and y labels are hung off the MEASURED tight bounding boxes of the outer panels
(`fig.canvas.draw()` then `get_tightbbox`), because how far the tick labels reach depends on the
font, the type size and whether LaTeX is typesetting, so any fixed offset either overlaps them or
leaves a gap. The panel letters are dropped below the inward tick marks, whose length is read
from `xtick.major.size` and converted to axes fraction: there is an x tick at -5 um sitting
directly under a corner-hugging label. Those ticks are also shortened to 2 pt from the house
default, `--tick-size`, because these panels are images rather than plots and the ticks run
inward over the data on all four sides; at the default length the top-left pair still reached the
letter. The letter position is computed from whatever tick length is in force, so changing one
moves the other. And the panels carry axis limits 13% wider than the disk,
so the page-white corner is a real wedge that holds the glyph. Without that margin the disk is
inscribed to the frame, the corner is too thin, and the letter needs either a halo or a white
patch notched out of the data, both of which look worse than the margin.

**Layout.** AIP double-column width, 6.69 in, drawn at final printed size so `\includegraphics`
must apply no scale factor or the type drops under the 8 pt AIP floor. The figure HEIGHT is solved
from the margins so the axes boxes are exactly square; setting it by eye leaves `aspect='equal'`
letterboxing each panel inside a taller box, which no amount of hspace tuning removes. Axis labels
are shared, one x and one y for the whole grid, since every panel is the same contact zone on the
same scale. Row captions and the two shared labels are positioned from the real axes rectangles
after the layout is fixed. `savefig` passes `bbox_inches=None` on purpose: the house RC sets
`savefig.bbox='tight'`, which would re-crop the canvas and move those hand-placed labels.


### The close-packing ceiling for TCR-pMHC, and where our densities sit

An upper bound on the surface density of the complex follows from its in-plane cross-section and
a packing fraction. Both inputs are choices, so the bound is a bracket, not one number.

| in-plane cross-section | d (nm) | hexagonal, 0.9069 | random close, 0.82 |
|---|---|---|---|
| width of the anchored ectodomain | 5.0 | 46188 | 41762 |
| Kihara equivalent sphere | 8.83 | 14816 | **13397** |
| slab-equivalent diameter given to the functional | 13.27 | 6557 | 5929 |
| full span, complex lying flat | 15.0 | 5132 | 4640 |

**Quote 13400 per um^2.** The complex is 15 nm long and about 5 nm wide, anchored roughly
perpendicular to the membrane. If it were a rigid perpendicular rod the footprint would be the
width alone and the ceiling would be about 42000 per um^2, which is an absolute limit no real
membrane approaches. The Kihara equivalent sphere is the orientationally averaged excluded
volume, it is what the exclusion in these runs is actually built on, and 13397 per um^2 at random
close packing is the honest ceiling for the object as modelled. The slab-equivalent row is not a
physical bound at all; it is the diameter handed to the three-dimensional functional to make a
cubic voxel represent an anchored monolayer, and it is listed only so the two are not confused.

Where the runs sit against that ceiling, peak density on 80 nm display cells at 200 s:

| | peak density (um^-2) | fraction of the 13397 ceiling | areal packing |
|---|---|---|---|
| reactive, point particles | 6406 | 0.48 | 0.37 |
| reactive, volume excluded | 4375 | 0.33 | 0.27 |
| transport only, point particles | 25000 | **1.87** | **1.53** |
| transport only, volume excluded | 7031 | 0.52 | 0.43 |

So the reactive model never goes unphysical: both arms stay below half the ceiling, and exclusion
is a real but moderate effect there. It is the transport-only arm, where every complex exists
from the start and nothing suppresses its accumulation, that drives the point-particle
description to 1.87 times the close-packing limit. For scale, physiological TCR is about 100 per
um^2 and Siokis uses 9 per um^2, so the initialization inherited from Isaacson & Zhang, 1410 per
um^2, is already an order of magnitude above the biology before any accumulation happens.


### Zero-force control, f0 = 0 (2026-09-10)

`F0_react_{point,excluded}`, canonical configuration otherwise, 200 s, both clean with zero
word-bound violations and a peak site occupancy of 17 of 64.

| t (s) | peak point (um^-2) | peak excluded | ratio |
|---|---|---|---|
| 2 | 2656 | 2500 | 1.06 |
| 10 | 2969 | 2500 | 1.19 |
| 100 | 2500 | 2344 | 1.07 |
| 200 | 2500 | 2500 | 1.00 |

With no ramp the complexes form wherever the reactants are, the core fills by diffusion alone, and
by 100 s the disk is uniform. Peak areal packing ends at 0.15 in both arms against 0.37 and 0.27
with the ramp. **The two arms are indistinguishable**, 1.00 at the final time, which is the control
the paper needs: exclusion does nothing measurable until the ramp builds a density for it to act
against. It also confirms the effect reported with the ramp is not an artefact of the exclusion
machinery being armed.

Note the reaction is *not* suppressed here. In the `db` gauge the binding equilibrium carries
exp(-beta Phi_C), so switching the ramp off removes that suppression and this arm forms more
complexes overall. The two figures are therefore not the same population redistributed.

★The two map figures carry DIFFERENT colour axes, 3906 against 1875 um^-2, because `--vmax`
defaults to the 99.5th percentile of the point arm. Rows within a figure are comparable; panels
across the two figures are not. Pass an explicit `--vmax` to both if they are ever placed side by
side.

### Colour

The ramp is NOT defined in this study. It belongs to the Kip3 microtubule-traffic study,
`../lmrddme_validation/kip3_traffic/kip3_palette.py`, and `synapse_maps_fig.py` imports `STOPS`
from it so the two applications in the paper share one scale. A local duplicate was written and
then deleted on 2026-09-10 when the owner pointed out the existing definition. If that study moves
the module, this raises ImportError rather than silently falling back to another palette.
Panel letters are off by default now (`--panel-letters 1` restores them).


### The schematic, `tcr_scheme_fig.py` (2026-09-10)

Three parts in one figure, no panel letters: the contact edge-on with the binding reaction, the
contact zone from above with the drive, and the potential the complex feels beside it on the same
radial axis. Laid out after Dharan & Farago Fig. 1 and using their geometry.

Panel one is drawn TO SCALE IN NANOMETRES with the axes locked to equal aspect, so the domain
cartoons keep their proportions and the 15 nm gap is really 15 nm. ★That means the x range is not
free: it is computed in `main()` from the axes-box ratio, so changing the figure height or the
axes rectangle without recomputing it shears every shape. The receptors are a key and a groove, not domain
diagrams: an earlier version drew the alpha-beta heterodimer and the class-I platform properly and
was more structural detail than a schematic of a lattice model can carry. ★The tab and the socket
are generated from ONE arc expression with a sign flip, so a bound pair nests exactly whenever the
two far edges meet, with no offset to keep in step if the sizes change. Bound pairs sit on a pale
capsule spanning exactly the cleft, which is what says the lattice carries the docked pair as one
species.

Four departures from their figure, all deliberate and all in the docstring: the binding is
explicit here and absent there; only the product is driven; there is no LFA1-ICAM1 and no pinning
obstacles; and the membranes are flat because this model has no membrane elasticity, so undulating
them would draw a mechanism the solver does not have.


### The combined application figure, `tcr_app_fig.py` (2026-09-10)

Built the way `kip3_app_fig.py` is: the drawing lives in the component scripts and the assembler
imports it, so the standalone figures and the combined one cannot drift. `synapse_maps_fig` was
refactored for this, lifting one panel's drawing into `draw_panel()` and the ramp into
`house_cmap()`; its own `main()` now calls both, so the standalone map figure is the same code.

Three blocks at 6.5 x 8.2 in: **A** the model, **B** driven at f0 = 1 kT/um, **C** the f0 = 0
control. Only the blocks are lettered; rows are named at the left and times are the column titles.

★**B and C share one colour axis**, taken from the driven point-particle arm. Separate axes would
stretch C to fill the bar and invite comparing a panel in B against one in C as though the scales
matched. C therefore sits low on the bar, which is the honest picture: peak 2500 against 6094 per
um^2, and the two arms in C agree to 1.00 where the driven arms differ by 1.39.

★The two drive rings on the disk are offset by half an angular step. Sharing the same eight
angles put every inner molecule on the ray its outer neighbour's arrow sweeps down, so the longer
outer arrows ran straight through them. The arrows are also ~2.3x longer than first drawn: the
1:2 length ratio was there before but neither arrow read as an arrow.

★The potential carries one complex per region with the arrow it feels, set in the clear space
INSIDE the V rather than on the curve, which leaves the arrows room at this size. What ties a
molecule to its region is the pair of guide lines it sits between, not contact with the branch,
and each arrow is short enough to stay inside its own region.

★Two things in the cartoon are sized in POINTS and had to become optional for the small panel:
the corner key naming the zone radii and the centre caption. At block-A size they spanned a third
of the disk. `draw_top(annotate=False)` drops both, and the radii are named by the guide lines in
the potential beside it. The side view's content is placed at FRACTIONS of its width for the same
reason: that axes is equal-aspect, so its x range is set by the axes-box ratio, and fixed
nanometre positions either overflow a short wide box or leave half of a tall one empty.


### Cartoon revisions (2026-09-11)

- **The contact zone is a flat circle.** A tilted 3D version was tried on 2026-09-11 and reverted
  the same day: the projection squashed the zone into a band, and the thing it was meant to fix,
  the two drive rings colliding, is already fixed by the half-step angular offset. What survives
  from that pass is the thinned scatter (14 per species in the application figure) and the longer
  arrows.
- **Bound pairs take the PRODUCT colour**, the same blue as the drive markers and the density
  panels, with the two halves differing only in tint so the interlock stays legible. The pale
  capsule is gone: the colour carries the "one species" statement on its own.
- ★**The f0 label was colliding with the curve, and it was not visible by eye.** Measured with a
  bbox-versus-polyline test: 44 curve points inside the label's bounding box, nearest approach
  0.0 pt. The potential axes is about 0.7 in tall for nine kT, so a 7 pt label is over one and a
  half kT high and anything set above the marker at r = 3 crosses the branch. Both labels now sit
  BESIDE their marker, at x positions chosen to miss the R_C and R_P guide lines; re-measured at
  7.7 and 22.5 pt clear. Keep that test in mind for any further edit to this panel.


### Vertical budget of the application figure

★A LaTeX float CANNOT break across pages: figure plus caption must fit the 665.5 pt text block, or
the caption prints off the paper with nothing reported by the build. That is the constraint that
sets `--height`, not taste. At 7.64 in the figure is 550 pt and the caption gets under five lines,
so the panel-by-panel description lives in the body text, which does flow across pages.

Spacing between blocks is paid for by narrowing the map columns (left 0.140, right 0.845), which
brings two rows of square panels from 2.03 in to 1.83 in and frees 0.4 in overall.

★The figure checks itself for text collisions after drawing and warns on stdout. It gathers axis
labels and titles as well as figure-level text: the first version looked only at `fig.texts` and
therefore passed while block C's banner sat on block B's x-axis label.


### Checked against the source model, Dharan & Farago 2017 (2026-09-07)

Isaacson & Zhang take their geometry, force profile and concentrations from Dharan & Farago,
*Soft Matter* **13**:6938 (2017), arXiv:1708.04414, which they describe as "only involved spatial
transport". Read directly, that model has TWO particle types and no chemistry:

- **Type A, TCR-pMHC bonds.** Mobile, pre-formed, N_A = 1000 for the whole contact area, on a
  lattice of spacing 100 nm with SINGLE OCCUPANCY, so the model already carries a hard core and a
  ceiling of 100 bonds per um^2. Cluster density about 80 per um^2. These are the only points
  that feel the active force. They also carry a nearest-neighbour attraction of -4.5 kT, the
  membrane-mediated potential of mean force.
- **Type B, membrane-cytoskeleton pinning proteins.** IMMOBILE, fixed for the whole run, average
  spacing 300 nm (500 nm changes little), and absent from the actin-depleted region r < 2 um.
  They feel no force. They are obstacles that type A must work around.
- **LFA1-ICAM1 is not modelled explicitly** in Dharan & Farago either. It enters only through the
  bond-length mismatch h0 = 27 nm used to derive the type A attraction.

Force: magnitude f0 for r < R_P and 2 f0 beyond, directed inward, exactly the profile implemented
here. **f0 = 0.1 pN**, which is 24.1 kT/um, and 2 f0 = 48.3 kT/um.

Three differences between what was run here and that source, none of them errors in the code:

| | Dharan & Farago | Isaacson & Zhang | run here |
|---|---|---|---|
| f0 | 0.1 pN = 24.1 kT/um | "f0 = 1", units unstated | 1 kT/um |
| complex density | about 80 per um^2 | 1410 per um^2 seeded | 1410 per um^2 |
| pinning obstacles | yes, immobile, none inside 2 um | absent | absent |

The force is the one that matters most. At 1 kT/um the drift is 0.060 um/s and a complex crosses
the disk in 93 s, which is why the runs here plateau near 150 s. At 0.1 pN the drift is 1.45 um/s
and the crossing takes 3.9 s. Isaacson & Zhang attribute their f0 = 1 to Siokis, whose couplings
are dimensionless weights, so the reading is genuinely ambiguous and worth running both ways.

The pinning points are the actionable gap: they are immobile excluded volume, they are absent
from the core by construction, and they are exactly the kind of object this method should carry.

### Both arms run at the Dharan-Farago force, 2026-09-07

f0 = 0.1 pN = 24.1 kT/um, transport only (their model has no chemistry), 100 s.

| run | outcome | core packing at 100 s | peak site occupancy |
|---|---|---|---|
| point particles, 1410 per um^2 | **CRASHED**, "unable to find an available site to handle a particle overflow" | - | >64 |
| volume excluded, 1410 per um^2 | ran clean | 0.679 | 60 |
| volume excluded + pins, 1410 per um^2 | ran clean | 0.592 | 58 |
| point particles, 80 per um^2 | overflow list on 1000 of every 1000 steps | - | >64 |
| volume excluded, 80 per um^2 | ran clean | 0.038 | 25 |

**At the source model's own force the point-particle description cannot be run at all.** It
overflows the lattice at Isaacson & Zhang's density and again at Dharan & Farago's own density,
seventeen times lower. Volume exclusion is not an improvement here, it is a precondition for the
simulation to exist. Every excluded arm ran clean with zero word-bound violations.

Caveat, stated rather than buried: the excluded arm at 1410 per um^2 reaches 60 of the 64
available slots, so the functional is at its clamp and the lattice cap is sharing the limiting
with it. The result is qualitatively safe and quantitatively capped.

**The pinning obstacles change nothing measurable.** At 300 nm spacing with 120 nm patches,
blocking 14.7% of sites outside the core, the core still half-fills at 2.0 s and reaches 90% at
4.0 s, identical to the run without them. That matches Dharan & Farago's own words, that the
bonds "individually navigate their way through the immobile pinning points", and it matches the
3.9 s ballistic crossing predicted from the force. Obstacles at this coverage slow nothing.

**The reaction shuts down completely at this force.** In the reactive arm only 122 complexes form
out of 121000 possible, against 36092 at f0 = 1 kT/um. The db gauge suppresses binding by
exp(-beta Phi_C), and Phi_C now spans 174 kT. So Isaacson & Zhang's two commitments, detailed
balance of reaction fluxes and the active force on the complex alone, are not merely
consequential at the physical force, they are incompatible with the model forming any product.

### What the pinning proteins actually are, and their size

Dharan & Farago cite Kusumi's picket-fence review for the 300 nm spacing, so the object is an
actin-anchored transmembrane picket and the 300 nm is a compartment size, not a molecular one.
The blocked area depends entirely on what you decide the obstacle is:

| representation | diameter | area blocked at 11.1 per um^2 |
|---|---|---|
| the picket protein itself, in plane | ~8 nm | 0.0006 |
| smallest obstacle a 40 nm lattice can hold | 40 nm | 0.014 |
| an actin fence around a 300 nm compartment, 7 nm filament | line | 0.047 |
| Dharan & Farago's blocked cell | 100 nm | 0.111 |
| our 3-voxel patch, as run | 120 nm | 0.147 |

Their point obstacle blocks 199 times the area of the protein it names. Modelling the protein
explicitly would delete the obstruction entirely. What their obstacle really coarse-grains is a
compartment boundary, and a fence lands within a factor of two of their value, so their choice is
defensible as a fence proxy and indefensible as a molecule.

### The deeper point about their exclusion

Their single-occupancy rule is not steric. The lattice spacing is 100 nm because that is the
range of the membrane-mediated potential of mean force, repulsive below 100 nm and attractive at
100 nm with depth 4.5 kT. Their cSMAC of 1000 bonds filling a 2 um core is 80 bonds per um^2,
which at the true 8.8 nm equivalent diameter is an areal packing fraction of 0.005. Sterically
that is empty. Confirmed here: the excluded run at 80 per um^2 ends at core packing 0.038 and is
indistinguishable from the point-particle result wherever the latter can run.

So Dharan & Farago's exclusion has a range of 100 nm and ours has a range of 9 nm. They are not
the same physics. Reproducing theirs means a lattice gas at h = 100 nm with the complex diameter
set equal to h, which is exactly the construction the microtubule application already uses. That
is the run that would actually reproduce their figure, and it has not been done.

### Still open
- LFA-1-ICAM-1 and CD45 are implemented and unused. Adding them is the second excluded species
  Isaacson & Zhang had to delete, and it is the obvious next arm.
- The physiological density (100 per um^2) and the Siokis density (9 per um^2) have not been run.
  At those the core areal fraction is a few percent and exclusion should be marginal; that
  contrast is worth having explicitly.
- The `--packing-match` diameter convention has not been run as a sensitivity arm.
