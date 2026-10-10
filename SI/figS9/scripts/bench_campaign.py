"""Cross-engine benchmark campaign driver (openspec: add-cross-engine-benchmark, task group 3).

Runs every (sweep point x engine x repetition) that does not already have an output file, so the
campaign is restartable. Engines:
  gpu_excl / gpu_ideal : vex_reactive.py --bench in the lm2.6 container on one B200
  cpu1 / cpu56         : cpu_drift_rdme (this directory), 1 thread pinned / 56 cores one socket
  readdy               : bench_readdy.py, SingleCPU, pinned to one core
Two-point differencing: the CPU and ReaDDy drivers difference internally; the GPU engine is run
at two step counts per repetition and differenced at analysis time.

Points (identical box and N across engines at each point):
  sweep 1  fixed volume 320x80x80 nm, n0 in {512, 1024, 2048, 4096, 8192}   (eta 0.007..0.113)
  sweep 2  fixed n0 = 2048, boxes (32,8,8) (32,16,16) (64,64,64) (128,128,128) sites
  sweep 3  fixed concentration 3.125e-3 /nm^3/species (the published value), n0 = 3.125*M
ReaDDy is truncated at sweep-3 boxes above 640^3 nm (n0 > 1e6) per the design's feasibility rule;
the truncation is recorded, not extrapolated.

Perf gates (2026-09-02): the drift arms run with the byte-identical word/transport bound and species
memo ARMED (VEX_WORD_BOUND=1 VEX_SPECIES_MEMO=1), i.e. the production setting of exp_coop_ec.py.
vex_reactive.py defaults them OFF, so the first campaign (kept in runs_pregates/) measured the drift
kernels moving all 16 words of every site; that cost 3-4x at dilute occupancy and was not physics.
Each JSON records `gates`. BENCH_GATES=0 reproduces the ungated arms. Every GPU run has a wall-clock
timeout (BENCH_TIMEOUT_S, default 900): a hung arm is logged as FAIL and retried on the next pass,
not allowed to stall the campaign.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs')
FTSZ = os.path.abspath(os.path.join(HERE, '..', '..', '..'))      # ftsz_tests: holds Lattice_Microbes_2.6/ and studies/
LM = os.path.join(FTSZ, 'Lattice_Microbes_2.6')
GPU_DEV = os.environ.get('BENCH_GPU', '0')
PIN_CORE = os.environ.get('BENCH_CORE', '40')
SOCKET_CORES = os.environ.get('BENCH_SOCKET', '0-55')     # one full socket, physical cores
REPS = 3
LOAD_MAX = float(os.environ.get('BENCH_LOAD_MAX', '30'))
WINDOW_TOL = 0.10
GATES = int(os.environ.get('BENCH_GATES', '1'))
TIMEOUT_S = float(os.environ.get('BENCH_TIMEOUT_S', '900'))


def quiesce():
    """Contention guard (spec: cross-engine-benchmark / timing protocol). Waits until the
    1-minute load is below LOAD_MAX and the benchmark GPU is idle; returns the load so it is
    recorded with the measurement."""
    while True:
        load = os.getloadavg()[0]
        gpu = subprocess.run(['nvidia-smi', '-i', GPU_DEV, '--query-gpu=utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True)
        busy = gpu.returncode == 0 and gpu.stdout.strip() and int(gpu.stdout.split()[0]) > 5
        if load < LOAD_MAX and not busy:
            return load
        time.sleep(20)


def windows_ok(rec):
    """Per-rep sanity only: the differenced estimate must be positive. The first window
    legitimately carries one-time costs (ReaDDy lazy neighbor-list build, GPU lattice
    allocation/upload growing with box volume) that the differencing exists to cancel;
    contention is caught ACROSS repetitions in vet_existing."""
    try:
        s1, s2 = rec['steps1'], rec['steps2']
        return (rec['wall2_s'] - rec['wall1_s']) / (s2 - s1) > 0
    except Exception:
        return False


def vet_existing():
    """Cross-repetition consistency: the differenced per-step estimates of one point must
    agree with their median within WINDOW_TOL; outlier reps are discarded for re-run."""
    import glob
    bad, groups = [], {}
    for f in sorted(glob.glob(os.path.join(OUT, '*.json'))):
        try:
            rec = json.load(open(f))
        except Exception:
            bad.append(f); continue
        if not windows_ok(rec):
            bad.append(f); continue
        per = (rec['wall2_s'] - rec['wall1_s']) / (rec['steps2'] - rec['steps1'])
        key = (rec['engine'], str(rec.get('dims', rec.get('box', []))), rec['n0'])
        groups.setdefault(key, []).append((f, per))
    for key, members in groups.items():
        if len(members) < REPS:
            continue
        vals = sorted(p for _, p in members)
        medv = vals[len(vals) // 2]
        for f, p in members:
            if medv > 0 and abs(p - medv) / medv > WINDOW_TOL:
                bad.append(f)
    for f in bad:
        print('VET: discarding inconsistent %s' % os.path.basename(f)); sys.stdout.flush()
        os.remove(f)
    return len(bad)

POINTS = []
for n0 in (512, 1024, 2048, 4096, 8192):
    POINTS.append(('s1', (32, 8, 8), n0))
for dims in ((32, 8, 8), (32, 16, 16), (64, 64, 64), (128, 128, 128)):
    POINTS.append(('s2', dims, 2048))
for dims in ((32, 8, 8), (32, 16, 16), (64, 64, 64), (128, 128, 128)):
    M = dims[0] * dims[1] * dims[2]
    POINTS.append(('s3', dims, int(round(3.125 * M))))
# dedupe identical (dims, n0) across sweeps but keep the tags for the figure
seen = {}
for sw, dims, n0 in POINTS:
    seen.setdefault((dims, n0), []).append(sw)
POINTS = [(tuple(tags), dims, n0) for (dims, n0), tags in seen.items()]


def steps_for(engine, dims, n0):
    """Two-point step counts sized so the long segment is seconds, not hours."""
    M = dims[0] * dims[1] * dims[2]
    if engine.startswith('gpu'):
        # dense (nper >= 1) AND large (>= 1e5 sites): per-step cost is occupancy-dominated
        # (~80 ms/step at 64^3, nper 3.125), so a 100/500 pair already times a multi-minute
        # segment. Dilute large boxes keep 500/2500; everything else the original 2000/10000.
        if n0 >= M and M >= 100000:
            return (100, 500)
        return (2000, 10000) if M <= 300000 else (500, 2500)
    if engine.startswith('cpu'):
        base = 3.0e-9 * M * 64 + 2.0e-7 * n0 * 6   # crude per-step model, seconds (1 thread)
        if engine == 'cpu56':
            base /= 20.0
        tgt = max(50, min(20000, int(30.0 / max(base, 1e-6))))
        return (max(10, tgt // 5), tgt)
    per = 1.4e-5 * n0 + 1e-3                        # readdy ~14 us per particle-step observed
    tgt = max(30, min(5000, int(60.0 / max(per, 1e-6))))
    return (max(10, tgt // 5), tgt)


def run(cmd, log):
    with open(log, 'a') as f:
        f.write('+ ' + ' '.join(cmd) + '\n'); f.flush()
        try:
            return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            # docker run is the child; the container keeps running after the client dies, so name it
            # (see the gpu branch) and kill it explicitly.
            name = cmd[cmd.index('--name') + 1] if '--name' in cmd else None
            if name:
                subprocess.call(['docker', 'kill', name], stdout=f, stderr=subprocess.STDOUT)
            f.write('TIMEOUT after %.0f s\n' % TIMEOUT_S); f.flush()
            return 124


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ''
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, 'campaign.log')
    for tags, dims, n0 in POINTS:
        M = dims[0] * dims[1] * dims[2]
        nper = n0 / float(M)
        dtag = 'x'.join(str(d) for d in dims)
        box = ','.join(str(d * 10) for d in dims)
        for engine in ('gpu_excl', 'gpu_ideal', 'gpu_base', 'cpu1', 'cpu56', 'readdy'):
            if only and only not in engine:
                continue
            if engine == 'readdy' and n0 > 100_000:
                # design's feasibility rule: at the published concentration the per-particle cost
                # measured at eta 0.088-0.113 is ~100 us/step, so n0 = 819200 costs ~1 h/rep for
                # a 40-step timing. The ReaDDy weak-scaling curve is truncated here and the
                # caption discloses it; the trend is already fixed by the three smaller boxes.
                continue
            s1, s2 = steps_for(engine, dims, n0)
            for rep in range(REPS):
                tag = '%s_%s_n%d_r%d' % (engine, dtag, n0, rep)
                outf = os.path.join(OUT, tag + '.json')
                if os.path.exists(outf):
                    continue
                load_at_start = quiesce()
                t0 = time.time()
                if engine.startswith('gpu'):
                    recs = []
                    for steps in (s1, s2):
                        npz = os.path.join(OUT, '%s_st%d.npz' % (tag, steps))
                        gate_env = [] if (engine == 'gpu_base' or not GATES) else \
                            ['-e', 'VEX_WORD_BOUND=1', '-e', 'VEX_SPECIES_MEMO=1']
                        cmd = ['docker', 'run', '--rm', '--name', 'bench_%s_st%d' % (tag, steps),
                               '--gpus', 'device=%s' % GPU_DEV,
                               '-v', '%s:/lm' % LM, '-v', '%s:/ftsz' % FTSZ, '-v', '/raid/racda/campaigns:/raid/racda/campaigns', '-w', '/ftsz/studies/lmrddme_validation/mpd_crddme',
                               '-e', 'PYTHONPATH=/lm/build_container:/lm/build_container/pylm:'
                                     '/lm/src/lm_diagnostics'] + gate_env + [
                               'lm2.6:latest', 'python3', 'vex_reactive.py',
                               '--nper', repr(nper), '--dims', ','.join(map(str, dims)),
                               '--gamma', '0.3,0.3,0.3', '--wavelength', '80', '--K', '200',
                               '--c-off', '500000', '--steps', str(steps), '--bench',
                               '--seed', str(rep + 1),
                               '--out', '/ftsz/studies/lmrddme_validation/benchmark/runs/%s_st%d.npz' % (tag, steps)]
                        if engine == 'gpu_ideal':
                            cmd.insert(-2, '--no-exclusion')
                        if engine == 'gpu_base':
                            cmd.insert(-2, '--baseline-kernel')
                        if run(cmd, log):
                            print('FAIL', tag, steps); break
                        recs.append((steps, npz))
                    else:
                        import numpy as np
                        m1 = json.loads(str(np.load(recs[0][1], allow_pickle=True)['meta']))
                        m2 = json.loads(str(np.load(recs[1][1], allow_pickle=True)['meta']))
                        per = (m2['wall_solver_s'] - m1['wall_solver_s']) / (s2 - s1)
                        rec = dict(engine=engine, dims=list(dims), M=M, n0=n0, rep=rep,
                                   steps1=s1, steps2=s2, wall1_s=m1['wall_solver_s'],
                                   wall2_s=m2['wall_solver_s'], per_step_s=per,
                                   dt_s=m1['dt_s'], tau_scale=m1['tau_scale'],
                                   sim_per_wall=m1['dt_s'] * m1['tau_scale'] / per,
                                   gates=(0 if engine == 'gpu_base' else GATES),
                                   hardware='NVIDIA B200')
                        json.dump(rec, open(outf, 'w'))
                elif engine.startswith('cpu'):
                    threads = '1' if engine == 'cpu1' else '56'
                    pin = ['taskset', '-c', PIN_CORE if engine == 'cpu1' else SOCKET_CORES]
                    env = dict(os.environ, OMP_NUM_THREADS=threads, OMP_PLACES='cores')
                    cmd = pin + [os.path.join(HERE, 'cpu_drift_rdme'),
                                 '--dims', ','.join(map(str, dims)), '--n0', str(n0),
                                 '--gamma', '0.3', '--wavelength', '80', '--K', '200',
                                 '--c-off', '500000', '--seed', str(rep),
                                 '--steps1', str(s1), '--steps2', str(s2), '--out', outf]
                    with open(log, 'a') as f:
                        f.write('+ ' + ' '.join(cmd) + '\n'); f.flush()
                        if subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env):
                            print('FAIL', tag)
                    if os.path.exists(outf):
                        rec = json.load(open(outf))
                        rec.update(engine=engine, rep=rep, hardware='Xeon Platinum 8570')
                        json.dump(rec, open(outf, 'w'))
                else:
                    cmd = ['taskset', '-c', PIN_CORE,
                           '/raid/racda/opt/readdy_env/bin/python',
                           os.path.join(HERE, 'bench_readdy.py'), '--box', box,
                           '--n0', str(n0), '--seed', str(rep),
                           '--steps1', str(s1), '--steps2', str(s2),
                           '--out', outf.replace('.json', '.npz')]
                    env = dict(os.environ, CONDA_PREFIX='/raid/racda/opt/readdy_env')
                    with open(log, 'a') as f:
                        f.write('+ ' + ' '.join(cmd) + '\n'); f.flush()
                        if subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env):
                            print('FAIL', tag)
                    npzf = outf.replace('.json', '.npz')
                    if os.path.exists(npzf):
                        import numpy as np
                        rec = json.loads(str(np.load(npzf, allow_pickle=True)['meta']))
                        rec.update(engine='readdy', dims=list(dims), M=M, rep=rep,
                                   hardware='Xeon Platinum 8570 (1 core)')
                        json.dump(rec, open(outf, 'w'))
                if os.path.exists(outf):
                    rec = json.load(open(outf))
                    rec['loadavg_at_start'] = load_at_start
                    if not windows_ok(rec):
                        print('NEGATIVE DIFFERENCED RATE, discarding %s for re-run' % tag)
                        os.remove(outf)
                    else:
                        json.dump(rec, open(outf, 'w'))
                print('%-28s %6.1fs  tags=%s' % (tag, time.time() - t0, '+'.join(tags)))
                sys.stdout.flush()
    print('CAMPAIGN PASS COMPLETE')


if __name__ == '__main__':
    vet_existing()
    for attempt in range(3):
        main()
        if not vet_existing():
            break
    print('CAMPAIGN VERIFIED COMPLETE')
