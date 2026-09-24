"""
Stage 2 - mechanism diagnostic.

Runs the same model at two budgets (one in the fast mode, one in the slow mode)
with per-storage stats on, and compares WHERE the extra recomputation goes.

The question it answers: is the slow mode a few tensors being evicted and
rebuilt over and over (thrashing on a small set), or recomputation spread
thinly across the whole graph (cascading dependency chains)? Those are
different failure mechanisms and imply different fixes.

Usage:
  py stage2_diag.py <log> --good 0.265 --bad 0.264 --out results\\diag.json
"""
import argparse, json, math, sys, threading, time

from simrd.parse import parse_file
from simrd.runtime import RuntimeV2EagerOptimized, RematExceededError
from simrd.heuristic import Heuristic
from simrd.heuristic.dtr import DTR


def baseline(callback):
    rt = RuntimeV2EagerOptimized(math.inf, Heuristic(), stats=False, trace=False)
    callback(rt)
    s = rt.telemetry.summary
    return {'compute': s['model_compute'], 'memory': s['max_memory'],
            'bottleneck': s['bottleneck_memory']}


def profile(callback, budget, base_compute, overhead_limit=60.0):
    rt = RuntimeV2EagerOptimized(budget, DTR(), stats=True, trace=False,
                                 remat_limit=base_compute * (overhead_limit - 1))
    status, t0 = 'ok', time.time()
    try:
        callback(rt)
    except MemoryError:
        status = 'oom'
    except RematExceededError:
        status = 'thrashed'
    tel = rt.telemetry
    s = tel.summary

    rows = []
    for rid in tel.storage:
        g = lambda c: tel.get('storage', rid, c)
        rows.append({
            'id': rid, 'size': g('size'),
            'evicts': g('evict_count'),
            'direct': g('direct_remat_count'),
            'collateral': g('collateral_remat_count'),
        })

    tot_ev = sum(r['evicts'] for r in rows)
    tot_dir = sum(r['direct'] for r in rows)
    tot_col = sum(r['collateral'] for r in rows)
    touched = [r for r in rows if r['evicts'] > 0]
    top = sorted(rows, key=lambda r: -r['evicts'])[:10]
    top_share = (sum(r['evicts'] for r in top) / tot_ev) if tot_ev else 0.0

    return {
        'budget': budget, 'status': status, 'wall_s': round(time.time() - t0, 2),
        'overhead': ((s['model_compute'] + s['remat_compute']) / s['model_compute']
                     if status == 'ok' and s['model_compute'] else None),
        'storages': len(rows),
        'storages_evicted': len(touched),
        'total_evictions': tot_ev,
        'direct_remats': tot_dir,
        'collateral_remats': tot_col,
        'collateral_ratio': (tot_col / (tot_dir + tot_col)) if (tot_dir + tot_col) else 0.0,
        'evicts_per_evicted_storage': (tot_ev / len(touched)) if touched else 0.0,
        'top10_share_of_evictions': round(top_share, 4),
        'top10': [{'id': r['id'], 'size_mb': round(r['size'] / 1e6, 2),
                   'evicts': r['evicts'], 'direct': r['direct'],
                   'collateral': r['collateral']} for r in top],
        'heuristic_evals': s['heuristic_eval_count'],
    }


def line(tag, p):
    print('  {:5s} budget {:8.1f} MB  overhead {}  evictions {:>9,}  '
          'evicted-storages {:>6,}  evicts/storage {:8.1f}  '
          'collateral {:5.1%}  top10 {:5.1%}'.format(
              tag, p['budget'] / 1e6,
              '{:7.3f}x'.format(p['overhead']) if p['overhead'] else '     --',
              p['total_evictions'], p['storages_evicted'],
              p['evicts_per_evicted_storage'], p['collateral_ratio'],
              p['top10_share_of_evictions']), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--good', type=float, required=True)
    ap.add_argument('--bad', type=float, required=True)
    ap.add_argument('--overhead-limit', type=float, default=60.0)
    ap.add_argument('--has-start', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    with open(args.log) as f:
        graph = parse_file(f, start=args.has_start)
    callback = graph.get_closure()

    base = baseline(callback)
    print('baseline: compute={:.1f} ms  peak={:.1f} MB'.format(
        base['compute'] / 1e6, base['memory'] / 1e6))

    out = {'log': args.log, 'baseline': base}
    for tag, ratio in (('GOOD', args.good), ('BAD', args.bad)):
        p = profile(callback, int(base['memory'] * ratio), base['compute'],
                    args.overhead_limit)
        p['ratio'] = ratio
        out[tag.lower()] = p
        line(tag, p)

    g, b = out['good'], out['bad']
    if g['total_evictions']:
        print('\n  evictions   BAD/GOOD = {:.1f}x'.format(
            b['total_evictions'] / g['total_evictions']))
    if g['storages_evicted']:
        print('  distinct storages evicted  BAD/GOOD = {:.2f}x'.format(
            b['storages_evicted'] / g['storages_evicted']))
    print('\n  If evictions blow up while distinct storages stays flat, the slow')
    print('  mode is thrashing a small set. If both scale together, it is a')
    print('  cascade spreading through the graph.')

    if args.out:
        import os
        d = os.path.dirname(args.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print('\nwrote', args.out)


if __name__ == '__main__':
    sys.setrecursionlimit(1000000)
    threading.stack_size(64 * 1024 * 1024)
    t = threading.Thread(target=main)
    t.start()
    t.join()
