# Kip3 traffic on MPD-RDDME, after Leduc et al. 2012

Working spec (parameters, mapping, figure plan): the *Protofilament Traffic* plan page, rev 2
(10 Sep 2026). This file is the run journal.

## Files

| file | role |
|---|---|
| `kip3_model.py` | constants, Leduc's parameter sets, the dimensionless → rate mapping, the single-sphere insertion work |
| `kip3_run.py` | builds and runs one arm (traffic / passing / attach / dilute), reduces frames to an npz |
| `kip3_gates.py` | scores the gate runs and prints traffic-run summaries |
| `lmrun.sh`, `lmrun_bg.sh` | container launchers (copies of the synapse study's; same private jLM) |
| `out/` | → `/raid/racda/campaigns/lmrddme_validation/kip3_traffic` |

## Two kernel facts the model is built around (verified in source, 10 Sep)

1. `computeFNAcceptance` uses the TOTAL energy change including the one-body field. A birth
   `∅ → M` under the tilt −αs would see ~10⁴ kT along the track. Hence the plus-end gauge
   Φ(s) = α(L−1−s) ≥ 0 (detachment `M → ∅` then has ΔΦ ≤ 0 everywhere and runs at exactly k_off)
   and attachment as `C → M` from an immobile, functional-invisible reservoir species C with the
   same coupling as M (tilt term cancels exactly). C is consumed, never created; depletion is
   recorded (`cres`). 48 copies per shell voxel.
2. A lone σ = h sphere carries βF_ex = 5.56 kT (σ = 9.99 nm) in a creating reaction (hops cancel
   it). Attachment into an empty site is accepted with e^{−5.56}; into an occupied one with e^{−50}.
   The uniform factor is absorbed into the nominal rate in the exclusion arm (×260).

Also: jLM's `run(seed=)` calls `lm.simulationFile`, which this build does not export; the seed
is stamped with `lm.SimulationFile(path).setParameter('seed', ...)` after `finalize()`.

## Gates, 10 Sep 2026 (all on the 1056×8×8 box, 1000-site track, q = 0.005, Δt = 8.93 ms)

| gate | arm | result |
|---|---|---|
| passing (two adjacent motors, 60 s) | excluded | never shared a voxel; gap min 1 — **exclusion holds** |
| passing | point | shared a voxel in 2 of 61 frames (control behaves) |
| dilute speed (160 tracers, 60 s) | excluded | **56.08 nm/s vs 56.0 target**, 0 backward links of 7876. First scoring gave 46.7 = 5/6·56 because 3 of 10 tracers per lane park at the tip; links within 10 sites of the tip are now dropped |
| dilute with switching | excluded | 57.9 nm/s by count-linking (unreliable when tracers change lanes); lane changes ≥ 0.25 /s per tracer, a lower bound against 0.60 |
| attach into a full lane (fig3d_hi rates, 200 s) | excluded | lane 0 stays at **exactly 1000** motors; empty lanes fill with k_on = 5.574e-3 vs 5.6e-3 (−0.5 %) — the ×260 absorption is right; C depleted 1.3 % |
| attach, empty track, 15 s | point | k_on from the count slope 5.79e-3 vs 5.6e-3 (+3.4 %, 1.3σ); with k_end = 0 the point arm overflows the tip within 200 s, as it must |

Cost: 103 µs/step without the reservoir, ~280 µs/step with it (13 occupancy words per shell
voxel). A 600 s traffic run is ~19 s of GPU.

## Fig. 1C runs (k̄on N = 0.24, ρ̄s = 0.91, k̄end = 0.2; 600 s; frames every 2 s)

`out/fig1c_exc_iso.npz`, `out/fig1c_exc_sw.npz`, `out/fig1c_pt_iso.npz`. Scores below.

### Results, 10 Sep 2026 (single seed each)

| run | jam length at 1500 s | ρ first 500 sites | ρ last 100 sites |
|---|---|---|---|
| point particles | 0 | 0.058 | 0.215 |
| exclusion, isolated lanes | 2.45 µm | 0.059 | 0.814 |
| exclusion, lane switching | 2.36 µm | 0.060 | 0.808 |
| 1-D TASEP-LK reference (parallel update, same Δt and p₊) | 2.68 µm | 0.061 | 0.799 |

Jam nucleates at the plus end at ~200 s and grows back (Leduc: kymograph Fig. 1B/C, growth Fig. 2);
upstream the density is the linear "antenna" ramp. Lane switching at Kip3's rate changes the jam
by 4 %, within what one seed resolves. Reference-vs-lattice difference in jam length (9 %) is
also a one-seed number; seeds are the next thing to add before the paper quotes it.
Figure: `out/fig_fig1c.{pdf,png}` (kymographs a–c, profiles d, jam length e). The 1-D reference is NOT drawn (user request, 10 Sep); its numbers stay in this table and `kip3_analyze.py --ref` still accepts it.

### Speed–density, Leduc Fig. 3D-high (k̄on N = 1, ρ̄s = 0.5, k̄end = 0.2; 600 s, 1 s frames)

`out/fig3dhi_exc_iso.npz`, `out/fig3dhi_pt_iso.npz`, `out/ref_fig3dhi.npz`; figure `out/fig_speed`.
Tracer speed against the local density of other motors (±10 sites, one-to-one linking, ~40k links
per run). Excluded lattice vs 1-D reference, v/v0 by ρ bin (0.05 … 0.95):
lattice 0.96 0.87 0.75 0.66 0.58 0.46 0.32 0.24 0.18 0.09; reference 0.97 0.87 0.75 0.66 0.58 0.46
0.33 0.25 0.18 0.09 — agreement to ±0.01 everywhere, on the mean-field 1−ρ line. Point particles
1.00 … 0.98 (0.92 in the last bin, the capped tip). This is Leduc's Fig. 3C/D.

### Open

- Seeds (3×) for jam length and the profile; the 9 % lattice-vs-reference gap at 1500 s.
- Fig. 5 phase diagram sweep (ρ̄s × k̄end at k̄on N = 1), both arms; ~100 runs at ~20–50 s each.
- Sidestepping effect at a parameter point where it should matter (lower k̄end, bottleneck jams).
- Transient double occupancy: `maxocc` = 2 appears in the excluded arms (opposite-direction hops
  into one site within one axis pass; backward hops are e^{-10}); count the frames before quoting.
- The 1-D reference omits backward steps and uses parallel update; Leduc's random-sequential
  update with p₊ = 1 differs at O(p₊) = 5 % in current — say so if the reference is quoted.

### The slowdown, and the sign of the density waves (10 Sep, `out/fig_slowdown`)

Asked whether the kymographs show a motor slowdown, and what the leaning features in the dense
region are. Measured on the Fig. 1C runs, frames from t = 1200 s (jam established), tracer links
per lane, linking window 40 sites: `kip3_analyze.py slowdown --window 40 --t-min 1200`.

| region | rho | motor v (tracers) | mean field (1-rho) v0 | density-wave v | v0(1-2rho) |
|---|---|---|---|---|---|
| exclusion, dilute (sites 100-600) | 0.086 | 50.8 nm/s = 0.91 v0 | 0.914 v0 | +48.4 nm/s | +46.4 |
| exclusion, jam (sites 820-990) | 0.803 | **11.4 nm/s = 0.20 v0** | 0.197 v0 | **-38.8 nm/s** | -33.9 |
| point particles, dilute / jam-position | 0.080 / 0.209 | 55.9 / 55.6 = **1.00 / 0.99 v0** | (no blocking) | +51.6 / +54.6 | - |
| lane switching, dilute / jam | 0.086 / 0.806 | 53.9 / 13.6 = 0.96 / 0.24 v0 | - | +43.7 / -35.7 | +46.4 / -34.3 |

Three distinct velocities, and they are not the same object:
- **motors**: slow from 0.91 v0 to 0.20 v0 across the wall, abruptly (panel a; the drop spans
  ~2 bins = 0.8 um and coincides with the density step), and both values sit on the mean-field
  (1-rho) v0 to within 2 %. This is Leduc's Fig. 3A. Point particles do not slow at all: 1.00 v0
  on both sides of a position where the excluded arm has dropped five-fold.
- **density waves**: the striations a kymograph shows. Forward at ~v0 in the dilute region,
  **backward at -39 nm/s inside the jam**, tracking the collective velocity v0(1-2rho) of an
  exclusion process, which changes sign at rho = 1/2 (panel b, from the cross-correlation of the
  lane-averaged density between consecutive frames). No motor ever steps backward
  (backward/forward = e^-10): the PATTERN moves backward while every particle moves forward.
  The point-particle arm has no such branch - its waves are forward at every density, because
  without exclusion the collective velocity is v0 independent of rho.
- **domain wall**: recedes from the plus end at -0.60 nm/s at late times (-1.9 nm/s averaged over
  the whole growth), set by mass balance across the shock, not by either of the above.

Three measurement traps found and fixed while doing this:
- The wave-velocity window must have the STATIC profile subtracted (a per-frame mean is not
  enough): a window holding the wall or the antenna ramp correlates on the immobile structure and
  pins the peak to zero shift. Symptom was a row of exact zeros in panel (b) at rho 0.3-0.8.
- The density reported next to a wave velocity must be averaged over the SAME late frames; over
  the whole run a jam window reads 0.60 instead of 0.80.
- ★The tracer linking window must exceed the per-frame travel. At 2 s frames a motor makes 11.2
  steps per frame (Poisson, sd 3.3), so a 15-site window silently drops the ~11 % of links that
  run further and truncates the displacement distribution from above: the point arm read 0.94 v0
  instead of 1.00, and the excluded dilute region 0.87 instead of 0.91. Window 40 fixes it. The
  Fig. 3D speed-density runs (1 s frames, mean 5.6 steps) were never affected.

Caveat on the lane-switching row: linking is done WITHIN a lane, so a tracer that sidesteps is
dropped and, worse, an arriving tracer can be matched to a departed one's earlier position. The
0.96 / 0.24 values are in the direction routing-around would give, but this tracker cannot
establish it - that needs lane-change-aware linking (open item).

### Configurations, trajectories and normalized density (`kip3_config_fig.py`, `out/fig_config`)

New single-purpose script (kip3_analyze.py was at 340 lines; this keeps both under the 500 cap).
Panels: (a)-(d) the configuration of the tube at t = 0, 400, 800, 1200 s — 16 lane rows against
position, one mark per occupied site, so a single motor is one dot and the jam is a solid block;
(e)-(g) the three kymographs with the snapshot times marked and single-tracer trajectories drawn
on the excluded arm; (h) profiles at the same times; (i) v/v0 against position at late times.

★ρ here is the **normalized motor density**: the per-frame profile divided by its mean along the
track, the same normalization `directed_transport.ipynb` used (`hist / hist.sum(...)`). 1 is the
track average, so the point-particle and excluded arms are comparable on one colour axis even
though their absolute occupancies differ four-fold. `--rho site` gives motors per binding site,
which is the quantity mean-field (1-rho) is a statement about, and is what panel (i)'s speeds are
scored against. The jam reads ~3.5 and the antenna region ~0.5 in normalized units.

The trajectories are the direct answer to "do the motors slow down": each one runs shallow through
the dilute region and kinks sharply where it enters the jam. Two traps in drawing them:
- ★Selecting the LONGEST tracks selects only late ones — the long tracks are those that end up
  parked in the jam, while a tracer arriving early walks to the tip and leaves within ~200 s.
  `sample_tracks` takes tracks of >= 20 frames spread evenly over start time instead.
- Linking is within a lane (see the switching caveat above), so the trajectory overlay is drawn on
  the isolated-lane arm only.

Motors bound at 0/400/800/1200 s: excluded 0/2752/3641/4131, switching 0/2752/3652/4106,
point 0/1809/1849/1899 — the point arm saturates at a third fewer motors because it has no jam to
hold them, and the excluded arm is still filling at 1200 s.

### Naming, the density scale, and the 3-D snapshot (10 Sep, second pass)

★Arms are now called **"no sidestepping"** and **"with sidestepping"** (was "isolated lanes" /
"lane switching") everywhere in `kip3_analyze.py`'s LABEL/SHORT, which every figure reads.

★**Normalized motor density rho runs 0 to 1**: motors per binding site, normalized to the
one-motor-per-site saturation that hard-core exclusion imposes. That is the quantity mean-field
(1 - rho) is a statement about, that Leduc's Langmuir density is quoted in, and that panel (i)'s
speeds are scored against, and it needs no rescaling to be comparable between arms. Point
particles have no ceiling and reach 1.5 motors per voxel at the plus end, which saturates the
colour axis - that overshoot is the argument, not an artefact. Two earlier definitions were tried
and rejected: per-frame mean normalization (`rho/<rho>`, which ran to 4 and hid the absolute
scale) and per-run max normalization (`--rho runmax`, still available, but the point arm's max is
its narrow tip spike so its lattice reads 0.13 and the panel goes black).

`kip3_snapshot3d.py` -> `out/fig_snap3d`: three 400-nm windows of the same tube at one instant -
dilute antenna region, the domain wall (located from the profile, not assumed), and inside the jam
- drawn in orthographic 3-D. The microtubule is a shaded cylinder because that is what the model
stands for; the 16 shell voxels are drawn literally as the wire cage, one cell per binding site
per lane; motors are beads at their true voxel centres, diameter 0.84 h so they read as separate
objects (the hard-sphere diameter is 0.999 h, which would have neighbours touching). Tracers are
a second colour, after the two-colour assay. Panel (d) is the lattice cross-section, which doubles
as the geometry figure the manuscript needs. Implementation notes: one depth scale for every
element (cylinder quads, cage lines, beads) so occlusion is consistent; back-facing cylinder quads
culled; beads are three nested vector circles stepped toward the light, so the PDF stays vector
and small; ambient 0.46, below which the tube's lower flank goes black; a faint 13-fold shading
modulation stands for the protofilaments. Renders in ~11 s.

All figures: jcp_style (AIP sizes, >= 8 pt at final size), drawn at final width, PDF + PNG.

### Colour scheme and the time-stacked 3-D figure (10 Sep, third pass)

`kip3_palette.py` holds the scheme, deep navy through blue and cyan to pale lime, and both figure
scripts import it. It is deliberately split in two: the continuous ramp (`P.cmap()`) is a monotone
lightness ramp and carries every density field, so it survives greyscale printing; the three
series colours (`P.ARM`: deep blue / teal / olive-lime) are chosen for SEPARATION rather than off
the ramp, differing in lightness (0.24 / 0.60 / 0.73) as well as hue, which is the property
jcp_style's own categorical set was checked for. Sampling markers off a continuous ramp by eye is
the trap here: two neighbouring stops are indistinguishable as points.

`kip3_snapshot3d.py --mode time` (default) is the snapshot sequence: one 720-nm window at the plus
end at t = 100, 250, 400, 1200 s, the slices staggered along the DEPTH axis so the series recedes,
earliest at the back and top, latest nearest the reader, with a time arrow down the left. rho in
the window runs 0.13 -> 0.44 -> 0.72 -> 0.81 as the jam front sweeps through it: 155 -> 936 motors.
`--mode space` keeps the earlier figure, three windows along the tube at one instant.
The scene is on the dark ground of the palette; `--dy` sets the slice separation (45 fills the
panel; at 30 the scene is wider than its axes box and a third of the panel is dead space).

Two implementation notes beyond the earlier ones: every slice shares ONE depth scale, so a slice
occludes the one behind it correctly; and `Figure` has no `annotate`, so the time arrow is an
`ax.annotate` with both xycoords and textcoords set to figure fraction.

### Fourth pass, 10 Sep: white ground, overview callout, speed panel

Per the owner, for publication:
- ★The rendered scenes are on **white**, and the scene colours in `kip3_palette.py` are tuned for
  it, NOT reused from the dark ground: the microtubule is a mid slate blue that survives shading
  down to ~40 % of base on its far flank, and motors are a deep teal so a 5 pt bead holds its edge
  against white. Putting the dark-ground values on white washes both out. The continuous ramp is
  unchanged (it is a field colormap, not a scene colour).
- Tracers are no longer drawn: one bead colour, and the legend lost its tracer entry. Tracers are
  still simulated - every speed measurement comes from them - just not rendered.
- The verbose titles are gone (no "the plus end, sites ...", no arm name on the render). Each
  slice carries only t and rho.
- ★Each slice now has the **whole 10-um microtubule beside it** - lane-averaged density along its
  full length, in the ramp - with the rendered window boxed and ruled across to the magnified
  view, so the blow-up is placed in the track it was cut from. The overview strips are drawn above
  the scene's zorder range so nothing in the render can cover them.
- ★The panel is sized from the scene's own aspect: the layout is computed before the figure is
  created (`geom`), then the figure height is set to the drawing's height plus a fixed 1.55 in
  band for the cross-section and legend. Guessing the height left a third of the panel empty,
  because an equal-aspect drawing floats inside a box of the wrong shape.
- `fig_fig1c` panel (e) is now the **speed plot** (v/v0 against the local density of other motors,
  with the mean-field 1-rho line, from the same runs, frames from `--speed-t-min` = 600 s) instead
  of jam length. The point-particle arm only populates the three lowest density bins - it never
  reaches high local density, which is the same statement the missing jam makes. Jam length is
  still computed and still printed by the script; it is only off the figure.

### Fifth pass, 10 Sep: keys off the data, and a cartoon of the two moves

- ★**Every key is now one shared legend row per FIGURE**, placed above the panel row, never inside
  an axes. Two things forced it: an in-axes legend eventually sits on data (the faint density
  profiles behind the speed panels were the casualty, and an opaque frame hides points rather than
  avoiding them), and the series labels are long enough that three of them crush each other when
  expanded across a half-width panel. `arm_handles` + `legend_row` in `kip3_analyze.py` build it;
  `kip3_config_fig.py` imports both. Panel letters and the time key moved to verified-empty
  corners inside each axes.
- The snapshot figure's cross-section is replaced by `draw_cartoon`: an **axial view** (looking
  down the tube: the 5x5 lattice, the 3x3 body with the tube inscribed, a motor in a shell cell and
  a curved arrow to the neighbouring lane = **sidestep**) beside a **side view** (the tube in
  profile with a row of binding sites, a motor and a straight arrow to the next site = **step**,
  with the plus-end direction marked). The axial view is the one the tube renders cannot give, and
  it is what makes the 16 lanes - and therefore the sidestep - visible at all.
- The "microtubule" and "voxel lattice" legend entries are gone, as are the "16 shell voxels" and
  "one motor, one voxel" annotations; the cartoon carries that information by construction. Only
  the motor key remains.

### Sixth pass, 10 Sep: the scheme figure

- `kip3_scheme_fig.py` -> `out/fig_scheme`: the model in four cartoons, now its own figure.
  (a) **axial view** — the 5x5 lattice, the 3x3 block the tube is inscribed in, the 16 shell
  voxels that are the lanes, and a double-headed arc for the **sidestep** (either way, same rate).
  (b) **side view** — the row of binding sites on the surface and the **step** to the next.
  (c) **the drive** — the lattice potential is a pure tilt, one step of -alpha per site, drawn as
  a staircase; the corrugation of the motor's own cycle is dashed BECAUSE IT IS NOT IN THE
  LATTICE: that barrier sits below the grid and is subsumed into k+, which is why k+ comes from
  the measured speed and not from a barrier height. The forward and backward arrows carry k+ and
  k+e^{-alpha}.
  (d) **birth-death** — attachment and detachment in one voxel: created at k_on per EMPTY site,
  removed at k_off (k_end at the plus end), and a second motor into an occupied voxel refused by
  the excluded-volume term, not by an imposed (1 - rho). That third state is the whole point of
  the panel.
- ★In panel (c) the motor is a PLOT MARKER, not a patch: that panel's axes are kT against site
  index, so a `Circle` in data coordinates comes out as a wide ellipse. The other panels are
  equal-aspect and use patches.
- The snapshot figure lost its cartoon and its motor legend; it is now only the time-stacked
  slices with their overviews, and is correspondingly shorter.
- `fig_fig1c` panel (d) is now speed AND density against position (what was `fig_slowdown` panel
  (a)) instead of the density profiles at several times; (e) is unchanged. The two panels needed
  wspace 0.46 - at 0.32 panel (d)'s twin-axis label collides with panel (e)'s y label.

### Seventh pass, 10 Sep: the scheme as ONE panel

- `kip3_scheme_fig.py` now draws everything in a **single axes with no sub-labels**, because the
  cartoon is meant to sit as one panel inside a larger figure. Its three parts are placed so each
  explains the next: the axial view top-left, the side view top-right, and **the potential
  directly beneath the side view on the same x**, with dotted guides tying the motor's binding
  site to its step on the staircase. The guides are drawn below the tube band's zorder, so they
  vanish behind it and reappear above and below - which is what makes the alignment read.
- Dropped at the owner's direction: the panel letters, "16 shell voxels = lanes", the expansion
  alpha = Delta_G_ATP/2 (now just alpha), and the "solid: the lattice / dashed: the motor's own
  cycle" caption. The dashed corrugation stays as the visual, unlabelled; a manuscript caption
  can carry what it is.
- `fig_fig1c`: both (d) and (e) labels moved to the upper left, both panels now use t >= 1200 s
  (`--speed-t-min` default), the mean-field 1-rho line is gone from (e) and from the shared key,
  and (e)'s x label is "local motor density rho".

### Eighth pass, 10 Sep

- The scheme's blocked-second-motor sub-cartoon is removed; the kinetics part is now just the
  birth-death pair, empty site ⇌ one motor, with k_on and k_off (k_end at the plus end). The
  exclusion argument is carried by the results, not by the cartoon.
- ★`fig_fig1c`: panel letters follow **the manuscript's own convention**, which the other figure
  scripts in this study folder already use - `text(0.0, 1.02, ..., ha='left', va='bottom')`, i.e.
  above the frame at the axes' LEFT EDGE (bench_fig.py, rxl80_occ_fig.py, fig_ribosome_arms.py;
  bd_figs.py uses -0.055 where a y label needs the room). At x = 0.0 the letter clears the tick
  labels, because those sit to the left of the axes rather than above it; a negative x is what
  puts a letter on top of them. The time window "t >= 1200 s" sits on the canvas in the top-RIGHT,
  the one strip both panels leave empty, and panel (e) carries ylim 1.27 to keep it clear of the
  topmost points.

### Into the manuscript, 10 Sep

★**The type did not match and the reason was silent.** `jcp_style.apply()` leaves `text.usetex`
off, so every label rendered in DejaVu Serif mathtext while the manuscript's own figures render in
Computer Modern. bench_fig.py, rxl80_occ_fig.py and fig_ribosome_arms.py all probe for `latex` and
switch usetex on; `kip3_analyze.style()` now does the same and warns loudly if latex is missing.
★**And the width was wrong.** The manuscript's single-column preprint text block is 6.5 in
(468 pt), not `jcp_style.FULL` = 6.69 in (the two-column reprint width). Drawn at 6.69 a figure is
scaled by 0.97 on the page and every 8 pt label falls under the AIP floor; fig_ribosome_arms.py
carries the same note. `kip3_app_fig.py` uses `TEXTW = 6.5`, and the schematic's annotations were
raised from 6.5 to 7 pt, the smallest size the manuscript's own figures use.

`kip3_app_fig.py` -> `out/fig_kip3_app` is THE manuscript figure, one figure in three lettered
blocks: **A** the cartoon (portrait layout), **B** the three kymographs with the speed against
local density beneath them, **C** the time-stacked configurations. Only the blocks are lettered -
the arms are named in the kymograph titles and in B's key. `kip3_snapshot3d.draw_time_stack` was
extracted so C is drawn into an axes of this figure rather than a figure of its own; running that
script standalone still works.

★Three layout rules this arrangement forced. The cartoon axes must end left of where the
kymographs' *y label* falls, not merely left of their frame (0.25 against a column at 0.355). In
the 'tall' cartoon the sidestep key goes BELOW the axial view - beside it the cartoon is 10.7
voxels wide and the key lands in the neighbouring panel's axis. And in C every slice's label is
anchored at the SAME x, the leftmost slice's, with the time arrow 16.5 voxels further left: the
per-slice anchors march left with the stagger, so the last one runs into the arrow, and 'far
enough left' is ~16 voxels at this scale, not the ~9 that looked right on paper.

The whole figure is 6.3 in tall: at 7.15 the caption ran off the page bottom, and block C is
width-limited so it loses nothing by the trim. B's key sits inside the speed panel's empty lower
left - above the panel it would fall between the kymographs' x label and the plot frame, with
room for neither.

Landed in `manuscripts/LMRDDME_updated/LM-RDDME`: `figures/pdf/kip3_traffic.pdf` (one figure),
a new subsection `sec:motors` after the synapse one, and three references (leduc2012jams,
parmeggiani2003, bugiel2015). `make` is clean - no undefined references or citations - the section
opens on p. 20 and the figure with its caption fills p. 22. NOT committed.

### Spacing pass on the merged figure (10 Sep)

- A: the sidestep key is just $k_s$ beside the arrow - no word, no rate. The rate belongs in the
  text, and at the cartoon's width a two-line key crowded whatever sat next to it.
- B: the key sits in the panel's **upper right, under the time note**. The lower-left corner looks
  empty but its last row reaches the points at $\rho = 0.45$; above $\rho = 0.55$ nothing rises
  past $v/v_0 = 0.35$, so that corner is the one genuinely free of data. Its labels are short
  forms ("point", "no sidestepping", "with sidestepping") because the kymograph titles overhead
  already carry the full names.
- C sits lower (axes bottom 0.005, height 0.365) so B's x label is not crowded by the block
  beneath it.
- ★The block letters are 13 pt and bold, and the bold had to go into the TeX: **under usetex
  matplotlib ignores `fontweight`**, so they are wrapped in `\textbf{}` when
  `rcParams['text.usetex']` is set and left plain when it is not.

- The sidestep rate is back, as two lines under the arc's LOWER head: `$k_s$ =` / `0.3 s^-1`,
  centred on the arrow at `cy - 2.7`, which puts it just below the cell block. To the right of
  the arc it would need 2.4 voxels the tall layout's width does not have, and level with it, it
  sits on the lattice. The axial view rides 0.4 voxels higher than before so the key clears the
  side view's plus-end arrow beneath it.
