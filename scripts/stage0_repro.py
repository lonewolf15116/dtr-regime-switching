"""
Stage 0 — reproduction harness.

Runs simrd (the DTR paper's own simulator) over a memory-budget sweep and
reports compute overhead relative to the unconstrained baseline.

Usage: PYTHONPATH=<simrd dir> python3 stage0_repro.py <log> [--heuristic DTR]
"""
import argparse, json, math, os, sys, time

from simrd.parse import parse_file
from simrd.runtime import RuntimeV1, RuntimeV2EagerOptimized, RematExceededError
from simrd.heuristic import Heuristic
from simrd.heuristic.dtr import DTR, DTREqClass, DTRLocal
from simrd.heuristic.etc import MSPS, LRU, LargestStorage, RandomStorage

HEURISTIC_MAP = {
    'DTR': DTR, 'DTREqClass': DTREqClass, 'DTRLocal': DTRLocal,
    'MSPS': MSPS, 'LRU': LRU, 'LargestStorage': LargestStorage,
    'RandomStorage': RandomStorage,
}

DEFAULT_RATIOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35,
                  0.3, 0.25, 0.2, 0.15, 0.1]


def baseline(callback):
    rt = RuntimeV1(math.inf, Heuristic(), stats=False, trace=False)
    callback(rt)
    s = rt.telemetry.summary
    assert s['remat_compute'] == 0, 'baseline should not rematerialize'
    return {
        'compute': s['model_compute'],
        'memory': s['max_memory'],
        'const': s['model_const_memory'],
        'bottleneck': s['bottleneck_memory'],
    }


def run_point(callback, budget, heuristic, overhead_limit=8.0, base_compute=None):
    kwargs = {'stats': False, 'trace': False}
    if base_compute is not None:
        kwargs['remat_limit'] = base_compute * (overhead_limit - 1)
    rt = RuntimeV2EagerOptimized(budget, heuristic, **kwargs)
    status, t0 = 'ok', time.time()
    try:
        callback(rt)
    except MemoryError:
        status = 'oom'
    except RematExceededError:
        status = 'thrashed'
    s = rt.telemetry.summary
    # NOTE: on oom/thrash the run aborted partway, so model_compute is a
    # partial total and the ratio is meaningless. Report None, never a number.
    valid = status == 'ok' and s['model_compute']
    return {
        'budget': budget,
        'status': status,
        'model_compute': s['model_compute'],
        'remat_compute': s['remat_compute'],
        'overhead': ((s['model_compute'] + s['remat_compute']) / s['model_compute']
                     if valid else None),
        'wall_s': round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--heuristic', default='DTR')
    ap.add_argument('--has-start', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.log) as f:
        graph = parse_file(f, start=args.has_start)
    callback = graph.get_closure()

    base = baseline(callback)
    print('baseline: compute={:.1f} ms  peak_mem={:.1f} MB  const={:.1f} MB  '
          'bottleneck={:.1f} MB'.format(
              base['compute'] / 1e6, base['memory'] / 1e6,
              base['const'] / 1e6, base['bottleneck'] / 1e6))

    h = HEURISTIC_MAP[args.heuristic]()
    rows = []
    for r in DEFAULT_RATIOS:
        budget = int(base['memory'] * r)
        row = run_point(callback, budget, h, base_compute=base['compute'])
        row['ratio'] = r
        rows.append(row)
        oh = '{:6.3f}x'.format(row['overhead']) if row['overhead'] else '     --'
        print('  ratio {:.2f}  budget {:7.1f} MB  overhead {}  {:9s} '
              '({:.1f}s)'.format(r, budget / 1e6, oh,
                                 row['status'], row['wall_s']), flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'log': args.log, 'heuristic': args.heuristic,
                       'baseline': base, 'points': rows}, f, indent=2)
        print('wrote', args.out)


if __name__ == '__main__':
    main()
