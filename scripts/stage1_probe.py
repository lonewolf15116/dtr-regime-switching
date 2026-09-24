"""
Stage 1 — probe harness.

Two jobs Stage 0 couldn't do:

  1. --ratios    fine-grained sweeps at ratios you name, to resolve anomalies
                 the coarse 0.05 grid only hints at.
  2. --bisect    binary-search the feasibility floor (lowest budget ratio that
                 still completes) to a stated tolerance, instead of reading it
                 off a coarse grid.

  --repeats N    re-runs each point N times. The DTR heuristic is deterministic,
                 so identical repeats mean an anomaly is a real property of the
                 policy, and differing repeats mean it is runtime nondeterminism.
                 That distinction decides whether an odd number is a finding or
                 a bug, so run it before believing any outlier.

Usage:
  py stage1_probe.py <log> --ratios 0.32,0.30,0.28,0.26,0.24,0.22 --repeats 3
  py stage1_probe.py <log> --bisect --tol 0.005
"""
import argparse, json, math, os, statistics, sys, threading, time

from simrd.parse import parse_file
from simrd.runtime import RuntimeV2EagerOptimized, RematExceededError
from simrd.heuristic import Heuristic
from simrd.heuristic.dtr import DTR, DTREqClass, DTRLocal
from simrd.heuristic.etc import MSPS, LRU, LargestStorage, RandomStorage

OVERHEAD_LIMIT = 8.0
HEURISTIC_MAP = {
    'DTR': DTR, 'DTREqClass': DTREqClass, 'DTRLocal': DTRLocal,
    'MSPS': MSPS, 'LRU': LRU, 'LargestStorage': LargestStorage,
    'RandomStorage': RandomStorage,
}


def baseline(callback):
    rt = RuntimeV2EagerOptimized(math.inf, Heuristic(), stats=False, trace=False)
    callback(rt)
    s = rt.telemetry.summary
    assert s['remat_compute'] == 0, 'baseline should not rematerialize'
    return {'compute': s['model_compute'], 'memory': s['max_memory'],
            'const': s['model_const_memory'], 'bottleneck': s['bottleneck_memory']}


def run_point(callback, budget, heuristic, base_compute, overhead_limit=None):
    overhead_limit = overhead_limit if overhead_limit else OVERHEAD_LIMIT
    rt = RuntimeV2EagerOptimized(
        budget, heuristic, stats=False, trace=False,
        remat_limit=base_compute * (overhead_limit - 1))
    status, t0 = 'ok', time.time()
    try:
        callback(rt)
    except MemoryError:
        status = 'oom'
    except RematExceededError:
        status = 'thrashed'
    s = rt.telemetry.summary
    valid = status == 'ok' and s['model_compute']
    return {
        'budget': budget, 'status': status,
        'overhead': ((s['model_compute'] + s['remat_compute']) / s['model_compute']
                     if valid else None),
        'remat_compute': s['remat_compute'],
        'wall_s': round(time.time() - t0, 2),
    }


def sweep(callback, base, h, ratios, repeats):
    rows = []
    for r in ratios:
        budget = int(base['memory'] * r)
        trials = [run_point(callback, budget, h, base['compute'])
                  for _ in range(repeats)]
        ohs = [t['overhead'] for t in trials if t['overhead'] is not None]
        stat = {t['status'] for t in trials}
        deterministic = len(set(ohs)) <= 1 and len(stat) == 1
        row = {'ratio': r, 'budget': budget, 'statuses': sorted(stat),
               'overheads': ohs, 'deterministic': deterministic,
               'mean_overhead': statistics.mean(ohs) if ohs else None,
               'wall_s': sum(t['wall_s'] for t in trials)}
        rows.append(row)
        oh = ('{:7.3f}x'.format(row['mean_overhead'])
              if row['mean_overhead'] else '      --')
        flag = '' if deterministic else '   <-- NONDETERMINISTIC'
        print('  ratio {:.3f}  budget {:8.1f} MB  overhead {}  {:<20s}'
              '({:.1f}s){}'.format(r, budget / 1e6, oh,
                                   ','.join(row['statuses']), row['wall_s'], flag),
              flush=True)
    return rows


def bisect_floor(callback, base, h, lo=0.05, hi=1.0, tol=0.005):
    """Lowest ratio that still completes. Assumes feasibility is monotone in
    budget; the LSTM result suggests verifying that rather than trusting it."""
    def feasible(r):
        out = run_point(callback, int(base['memory'] * r), h, base['compute'])
        print('    probe {:.4f} -> {}'.format(r, out['status']), flush=True)
        return out['status'] == 'ok'

    if feasible(lo):
        return {'floor': lo, 'note': 'feasible at search floor; try lower --lo'}
    if not feasible(hi):
        return {'floor': None, 'note': 'infeasible even at full budget'}
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return {'floor': hi, 'bracket': [lo, hi], 'tol': tol}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--heuristic', default='DTR')
    ap.add_argument('--ratios', default=None,
                    help='comma-separated, e.g. 0.32,0.30,0.28')
    ap.add_argument('--repeats', type=int, default=1)
    ap.add_argument('--bisect', action='store_true')
    ap.add_argument('--lo', type=float, default=0.05)
    ap.add_argument('--tol', type=float, default=0.005)
    ap.add_argument('--overhead-limit', type=float, default=8.0)
    ap.add_argument('--has-start', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    global OVERHEAD_LIMIT
    OVERHEAD_LIMIT = args.overhead_limit

    with open(args.log) as f:
        graph = parse_file(f, start=args.has_start)
    callback = graph.get_closure()

    base = baseline(callback)
    print('baseline: compute={:.1f} ms  peak_mem={:.1f} MB  bottleneck={:.1f} MB '
          '({:.1%} of peak)'.format(
              base['compute'] / 1e6, base['memory'] / 1e6,
              base['bottleneck'] / 1e6, base['bottleneck'] / base['memory']))

    h = HEURISTIC_MAP[args.heuristic]()
    result = {'log': args.log, 'heuristic': args.heuristic, 'baseline': base}

    if args.bisect:
        print('bisecting feasibility floor (tol={})...'.format(args.tol))
        result['bisect'] = bisect_floor(callback, base, h, args.lo, 1.0, args.tol)
        f = result['bisect']['floor']
        print('  floor = {}  ({:.2f}x bottleneck)'.format(
            f, (f * base['memory']) / base['bottleneck']) if f else '  no floor')

    if args.ratios:
        ratios = [float(x) for x in args.ratios.split(',')]
        print('sweeping {} ratios x {} repeats...'.format(len(ratios), args.repeats))
        result['points'] = sweep(callback, base, h, ratios, args.repeats)

    if args.out:
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print('wrote', args.out)


if __name__ == '__main__':
    sys.setrecursionlimit(1000000)
    threading.stack_size(256 * 1024 * 1024)
    t = threading.Thread(target=main)
    t.start()
    t.join()
