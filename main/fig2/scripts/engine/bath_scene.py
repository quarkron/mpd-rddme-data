"""Shared pieces of the bath drivers: stamp_msd.py (spherical tracers) and chain_bath.py (semiflexible chains).

A "scene" is one pyLM RDME simulation (the byte-lattice bath) plus a DriftMpdRdmeSolver whose structure handler
carries the bodies. Both drivers need the same three things, which live here so they cannot drift apart:

1. `BathRecorder`: a solver subclass whose hook re-arms, after `runSolver`'s init, whatever the init resets (reciprocal
   VEX, the sub-voxel hop divisor, the friction factor, diagnostics), then records the centre of mass of every complex
   on a frame grid, unwrapped for periodic boundaries. The grid is either a physical-time spacing (`frame_dt`) or, in
   bench mode, one frame per hook call (`frame_steps`: the hook interval IS the frame interval, so the frame count is
   exact and the physical clock is steps times tau).
2. `finish_sim`: the tail every scene shares: the engine's random seed, timestep, run length, write intervals, hook
   interval. Without `setRandomSeed` the engine seeds itself from the wall clock (MpdRdmeSolver.cu) and two same-seed
   runs diverge; this is where that is prevented.
3. `run_scene`: save (or reuse) the lattice file, run, and keep copies for the loop's identity gate: `save_init_lm`
   copies the freshly saved file BEFORE the run (the file a later run reuses, so every build starts from the same
   lattice and RNG stream); `keep_lm` copies the finished file AFTER the run (the lattice frames the gate hashes).
   A reuse path that does not exist aborts; nothing is rebuilt in its place.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import lm
from pyLM.units import microsecond


class BathRecorder(lm.DriftMpdRdmeSolver):
    """Records every complex's centre of mass per frame; `rearm(self)` runs once, on the first hook, after init."""

    def configure_recording(self, n_complexes, box_vox, frame_dt=None, frame_steps=None, rearm=None, extra=None,
                            progress_every=50):
        if (frame_dt is None) == (frame_steps is None):
            raise ValueError("configure_recording: give exactly one of frame_dt (physical grid) or frame_steps (bench)")
        self._n = int(n_complexes)
        self._box = np.asarray(box_vox, float)
        self._frame_dt = frame_dt
        self._next_t = 0.0                    # schedule against the NEXT TARGET, not elapsed-since-last,
        self._times = []                      # or the hook grid residual accumulates and inflates the cadence
        self._coms = []
        self._extras = []
        self._prev_raw = None
        self._shift = np.zeros((self._n, 3))
        self._rearm = rearm
        self._rearmed = False
        self._extra = extra                   # optional callable(self) -> array stored per frame (chain backbones)
        self._progress_every = progress_every
        self._t0_wall = time.perf_counter()

    def _read_coms(self):
        out = np.zeros((self._n, 3))
        o3 = np.zeros(3)
        for i in range(self._n):
            self.getStructureCOM(i, o3)
            out[i] = o3
        return out

    def hookSimulation(self, t, lattice):
        if not getattr(self, '_rearmed', False):
            # runSolver's init re-runs the structure-enable path, which resets struct_vex_reciprocal_ (and the other
            # armed switches) to their defaults. Anything armed before runSolver is silently dropped; with reciprocal
            # VEX lost, eta 0 and eta 0.15 gave bit-identical trajectories. The driver's rearm() puts them back here.
            if self._rearm is not None:
                self._rearm(self)
            self._rearmed = True
        tphys = float(t)
        if self._frame_dt is not None and tphys + 1e-15 < self._next_t:
            return 0                          # surplus hook call: cost is this comparison only
        raw = self._read_coms()
        if self._prev_raw is not None:        # unwrap: a jump past half the box is a periodic wrap
            d = raw - self._prev_raw
            self._shift -= self._box * np.round(d / self._box)
        self._prev_raw = raw
        self._times.append(tphys)
        self._coms.append(raw + self._shift)
        if self._extra is not None:
            self._extras.append(self._extra(self))
        if self._frame_dt is not None:
            self._next_t += self._frame_dt
        if self._progress_every and len(self._times) % self._progress_every == 0:
            print(f"    frame {len(self._times):5d}  t={tphys:.3f}s  wall={time.perf_counter()-self._t0_wall:.1f}s", flush=True)
        return 0                              # observation only, no lattice change

    def frames(self):
        """(times [F], coms [F, n, 3] voxels unwrapped, extras list) recorded so far."""
        return np.asarray(self._times), np.asarray(self._coms), list(self._extras)


NEVER_STEPS = 1000000000        # write interval "never": fits the engine's uint32 step counter


def finish_sim(sim, seed, tau_s, t_end, hook_steps, write_steps=NEVER_STEPS, lattice_write_steps_=NEVER_STEPS):
    """Seed, clock and cadence of a scene. The engine reads the write, lattice-write and hook intervals as STEP COUNTS
    (uint32), not seconds. Both write intervals default to 'never': the frames come from the hook. A finite lattice write
    interval is used only to keep lattice frames for the identity gate (lattice_write_steps)."""
    sim.setRandomSeed(int(seed))
    sim.setTimestep(microsecond(tau_s * 1e6))
    sim.setSimulationTime(float(t_end))
    sim.setWriteInterval(int(write_steps))
    sim.setLatticeWriteInterval(int(lattice_write_steps_))
    sim.setHookInterval(int(hook_steps))


def bench_plan(steps, frame_steps, tau_s):
    """Bench mode: run exactly `steps` timesteps with one frame every `frame_steps` (the hook interval)."""
    if steps <= 0 or frame_steps <= 0:
        raise ValueError("bench mode needs steps > 0 and frame_steps > 0")
    if steps % frame_steps != 0:
        raise ValueError(f"bench mode: steps ({steps}) must be a multiple of frame_steps ({frame_steps})")
    return float(steps) * float(tau_s), int(frame_steps)


def lattice_write_steps(t_end, tau_s, lattice_frames):
    """Lattice write interval IN TIMESTEPS that yields `lattice_frames` lattice dumps over the run (0 frames = never).
    The engine parses writeInterval / latticeWriteInterval / hookInterval as unsigned step counts (MpdRdmeSolver.cu,
    generateTrajectory): a fractional value truncates to 0 and means EVERY step, which is how a 500-step run once produced
    501 lattice frames and 400 MB."""
    if lattice_frames <= 0:
        return NEVER_STEPS
    total = int(round(float(t_end) / float(tau_s)))
    return max(1, total // int(lattice_frames))


def run_scene(sim, sol, gpu, reuse_lm=None, save_init_lm=None, keep_lm=None, tag='scene'):
    """Save or reuse the lattice file, run the solver, keep the requested copies. Returns the wall time in seconds."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'scene.lm')
        if reuse_lm:
            if not os.path.isfile(reuse_lm):
                sys.exit(f"[{tag}] --reuse-lm: {reuse_lm} does not exist; build it first with --save-init-lm")
            shutil.copyfile(reuse_lm, path)
            print(f"[{tag}] reusing prebuilt lattice {reuse_lm} ({os.path.getsize(reuse_lm) / 1e6:.1f} MB)", flush=True)
        else:
            sim.save(path)
        if save_init_lm:
            os.makedirs(os.path.dirname(os.path.abspath(save_init_lm)), exist_ok=True)
            shutil.copyfile(path, save_init_lm)
            print(f"[{tag}] saved initial lattice {save_init_lm}", flush=True)
        sol.validateStability()
        t0 = time.perf_counter()
        lm.runSolver(path, 1, sol, [int(gpu)], 0)
        wall = time.perf_counter() - t0
        if keep_lm:
            os.makedirs(os.path.dirname(os.path.abspath(keep_lm)), exist_ok=True)
            shutil.copyfile(path, keep_lm)
            print(f"[{tag}] kept finished lattice file {keep_lm} ({os.path.getsize(keep_lm) / 1e6:.1f} MB)", flush=True)
    return wall


def add_scene_args(ap):
    """The CLI options every scene driver shares (seed, bench, lattice reuse/keep)."""
    ap.add_argument('--seed', type=int, default=1, help='engine RNG seed (also the placement seed)')
    ap.add_argument('--bench', type=int, default=0, metavar='STEPS',
                    help='bench mode: run exactly STEPS timesteps with one frame every --frame-steps (overrides --t-end/--frame-dt)')
    ap.add_argument('--frame-steps', dest='frame_steps', type=int, default=0,
                    help='bench mode frame interval in steps (default: derived from --frame-dt)')
    ap.add_argument('--reuse-lm', dest='reuse_lm', default=None, help='start from this prebuilt lattice file (must exist)')
    ap.add_argument('--save-init-lm', dest='save_init_lm', default=None, help='copy the freshly built lattice file here BEFORE the run')
    ap.add_argument('--keep-lm', dest='keep_lm', default=None, help='copy the finished lattice file here AFTER the run (identity gate)')
    ap.add_argument('--lattice-frames', dest='lattice_frames', type=int, default=0,
                    help='write this many lattice frames into the lattice file over the run (0 = none; each 64^3 frame is 83 MB)')
