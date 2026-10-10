"""Campaign driver for the advantage-metric sweep. Restartable: every (point, arm, rep) that
already has an output file is skipped, so an interrupted run resumes by being re-invoked.

Six arms, chosen so that the cost ratio can be quoted at matched hardware and at each engine's
best kernel rather than only at the configuration the accuracy chapters forced:

  lat_gpu        DriftMpdRdmeSolver, one B200
  lat_cpu1       adv_cpu, one pinned core           matched physics, matched hardware vs rd_sc
  lat_cpu56      adv_cpu, one socket
  rd_sc          ReaDDy SingleCPU + DetailedBalance, one pinned core   MATCHED PHYSICS
  rd_cpu1        ReaDDy CPU kernel + Gillespie, one pinned core        BEST SINGLE-CORE KERNEL
  rd_cpu56       ReaDDy CPU kernel + Gillespie, one socket

usage: python adv_campaign.py [substring filter on arm name]
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adv_schemes as S

OUT = os.path.join(HERE, 'runs')
FTSZ = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
LM = os.path.join(FTSZ, 'Lattice_Microbes_2.6')
RUNS_ABS = '/raid/racda/campaigns/lmrddme_validation/advantage/runs'
GPU_DEV = os.environ.get('ADV_GPU', '6')
PIN_CORE = os.environ.get('ADV_CORE', '40')
SOCKET = os.environ.get('ADV_SOCKET', '0-55')
NTHREAD = int(os.environ.get('ADV_NTHREAD', '56'))
REPS = int(os.environ.get('ADV_REPS', '3'))
READDY_PY = '/raid/racda/opt/readdy_env/bin/python'
TIMEOUT_S = float(os.environ.get('ADV_TIMEOUT_S', '1800'))
SCHEMES = os.environ.get('ADV_SCHEMES', 'diff,rev,net').split(',')
TARGET_S = float(os.environ.get('ADV_TARGET_S', '10'))   # wall-clock target, long segment
# The reaction-count sweep (2026-09-30) runs the pr* schemes at one packing fraction and only the
# arms the figure plots: ADV_SCHEMES=pr0,...,pr6 ADV_ETAS=0.09 ADV_ARMS=lat_gpu,lat_cpu1,rd_sc,rd_cpu1.
ETAS = [float(x) for x in os.environ['ADV_ETAS'].split(',')] if os.environ.get('ADV_ETAS') else list(S.ETA_SWEEP)
ARMS = os.environ.get('ADV_ARMS', 'lat_gpu,lat_cpu1,lat_cpu56,rd_sc,rd_cpu1,rd_cpu56').split(',')


def quiesce():
    while True:
        load = os.getloadavg()[0]
        g = subprocess.run(['nvidia-smi', '-i', GPU_DEV, '--query-gpu=utilization.gpu',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
        busy = g.returncode == 0 and g.stdout.strip() and int(g.stdout.split()[0]) > 5
        if load < 120 and not busy:
            return load
        time.sleep(15)


def steps_for(arm, scheme, eta):
    """Two-point step counts sized so the long segment is ~TARGET_S, from a crude per-step model
    fitted to the probe runs. Over- or under-shooting costs wall time, not correctness."""
    n0 = sum(S.counts_for_eta(scheme, eta).values())
    nr = max(1, len(S.SCHEMES[scheme]['reactions']))
    if arm == 'lat_gpu':
        return (500, 2500)
    if arm.startswith('lat_cpu'):
        # Measured socket speedup is ~3x, not the 20x a core count suggests: 2048 subvolumes is
        # far too small a lattice to fill 56 cores. Over-estimating it only wastes wall time,
        # since the step count is sized to hit TARGET_S.
        per = (3.0e-9 * S.M_SUB * 64 + 2.0e-7 * n0 * 6) / (3.0 if arm.endswith('56') else 1.0)
    else:
        # ReaDDy: ~0.7 us per particle-step at k_off 200, superlinear in density; the reaction
        # handler adds with the channel count.
        per = 7.0e-7 * n0 * (1.0 + 4.0 * eta / 0.02) * (1.0 + 0.15 * nr)
        if arm == 'rd_cpu56':
            per /= 2.5        # measured, not assumed: the dilute point ran 2.4x faster on a socket
    tgt = max(40, min(20000, int(TARGET_S / max(per, 1e-7))))
    return (max(10, tgt // 5), tgt)


def run(cmd, log, name=None):
    with open(log, 'a') as f:
        f.write('+ ' + ' '.join(cmd) + '\n'); f.flush()
        try:
            return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            if name:
                subprocess.call(['docker', 'kill', name], stdout=f, stderr=subprocess.STDOUT)
            f.write('TIMEOUT after %.0f s\n' % TIMEOUT_S); f.flush()
            return 124


def gpu_cmd(scheme, eta, rep, outf, name):
    return ['docker', 'run', '--rm', '--name', name, '--gpus', 'device=%s' % GPU_DEV,
            '--user', '%d:%d' % (os.getuid(), os.getgid()), '-e', 'HOME=/tmp',
            '-v', '%s:/lm' % LM, '-v', '%s:/ftsz' % FTSZ,
            '-v', '/raid/racda/campaigns:/raid/racda/campaigns',
            '-w', '/ftsz/studies/lmrddme_validation/advantage',
            '-e', 'PYTHONPATH=/lm/build_container:/lm/build_container/pylm:/lm/src/lm_diagnostics',
            'lm2.6:latest', 'bash', '-lc',
            'source /opt/conda/etc/profile.d/conda.sh && conda activate lm_2.6 && '
            'python adv_lattice.py --scheme %s --eta %g --seed %d --gpu 0 '
            '--steps1 %d --steps2 %d --out %s'
            % (scheme, eta, rep, 500, 2500, outf.replace(OUT, RUNS_ABS))]


def cpu_cmd(scheme, eta, rep, outf, threads, s1, s2):
    sig = [S.SCHEMES[scheme]['sigma'][n] for n in S.SCHEMES[scheme]['species']]
    idx = {n: i for i, n in enumerate(S.SCHEMES[scheme]['species'])}
    counts = S.counts_for_eta(scheme, eta)
    seed_counts = ','.join('%d:%d' % (idx[n], c) for n, c in counts.items() if c)
    cmd = [os.path.join(HERE, 'adv_cpu'),
           '--dims', ','.join(map(str, S.DIMS)),
           '--nspec', str(len(sig)), '--sigma', ','.join('%.6g' % x for x in sig),
           '--gamma', ','.join(['%g' % S.GAMMA] * len(sig)),
           '--wavelength', '%g' % S.WAVELENGTH, '--dt', '%g' % S.DT_S,
           '--seed-counts', seed_counts, '--seed', str(rep),
           '--steps1', str(s1), '--steps2', str(s2), '--out', outf]
    for kind, react, prod, rate in S.rates(scheme, 'kernel'):
        lhs = ','.join(str(idx[x]) for x in react)
        rhs = ','.join(str(idx[x]) for x in prod)
        cmd += ['--rxn', '%s:%s/%s@%.10g' % (kind, lhs, rhs, rate)]
    pin = ['taskset', '-c', PIN_CORE if threads == 1 else SOCKET]
    return pin + cmd


def readdy_cmd(scheme, eta, rep, outf, arm, threads, s1, s2):
    pin = ['taskset', '-c', PIN_CORE if threads == 1 else SOCKET]
    return pin + [READDY_PY, os.path.join(HERE, 'adv_readdy.py'),
                  '--scheme', scheme, '--eta', '%g' % eta,
                  '--arm', 'sc_db' if arm == 'rd_sc' else 'cpu_g',
                  '--threads', str(threads), '--seed', str(rep),
                  '--steps1', str(s1), '--steps2', str(s2), '--out', outf]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ''
    os.makedirs(OUT, exist_ok=True)
    log = os.path.join(OUT, 'campaign.log')
    arms = ARMS
    todo = [(sc, eta, arm, rep) for sc in SCHEMES for eta in ETAS
            for arm in arms for rep in range(REPS)
            if not only or only in arm]
    done = 0
    for sc, eta, arm, rep in todo:
        tag = '%s_eta%.3f_%s_r%d' % (sc, eta, arm, rep)
        outf = os.path.join(OUT, tag + '.json')
        done += 1
        if os.path.exists(outf):
            continue
        s1, s2 = steps_for(arm, sc, eta)
        t0 = time.time()
        quiesce()
        if arm == 'lat_gpu':
            rc = run(gpu_cmd(sc, eta, rep, outf, 'adv_' + tag), log, 'adv_' + tag)
        elif arm.startswith('lat_cpu'):
            th = 1 if arm == 'lat_cpu1' else NTHREAD
            env = dict(os.environ, OMP_NUM_THREADS=str(th), OMP_PLACES='cores')
            with open(log, 'a') as f:
                cmd = cpu_cmd(sc, eta, rep, outf, th, s1, s2)
                f.write('+ ' + ' '.join(cmd) + '\n'); f.flush()
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                     timeout=TIMEOUT_S)
        else:
            th = 1 if arm in ('rd_sc', 'rd_cpu1') else NTHREAD
            env = dict(os.environ, CONDA_PREFIX='/raid/racda/opt/readdy_env')
            with open(log, 'a') as f:
                cmd = readdy_cmd(sc, eta, rep, outf, arm, th, s1, s2)
                f.write('+ ' + ' '.join(cmd) + '\n'); f.flush()
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                     timeout=TIMEOUT_S)
        ok = 'ok' if rc == 0 and os.path.exists(outf) else 'FAIL rc=%s' % rc
        print('[%3d/%3d] %-34s %-6s %5.0fs' % (done, len(todo), tag, ok, time.time() - t0))
        sys.stdout.flush()
    print('ADVANTAGE CAMPAIGN PASS COMPLETE %s' % time.strftime('%F %T'))


if __name__ == '__main__':
    main()
