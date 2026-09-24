import sys, threading, json, time, math
from collections import Counter, defaultdict
sys.setrecursionlimit(1000000)
from simrd.parse import parse_file
from simrd.runtime import RuntimeV2EagerOptimized, RematExceededError
from simrd.heuristic.dtr import DTR

LOG="/home/claude/dtr-prototype/simrd/logs/lstm-128-11000000000.0-2020-10-1-16-38-52-default.log"

def build():
    with open(LOG) as f:
        graph = parse_file(f, start=False)
    return graph.get_closure()

def baseline(cb):
    from simrd.heuristic import Heuristic
    rt = RuntimeV2EagerOptimized(math.inf, Heuristic(), stats=False, trace=False)
    cb(rt); s=rt.telemetry.summary
    return s['max_memory'], s['model_compute']

def run(ratio, peak, base_compute):
    cb = build()
    budget=int(peak*ratio)
    rt = RuntimeV2EagerOptimized(budget, DTR(), stats=False, trace=True,
                                 remat_limit=base_compute*(60-1))
    st='ok'; t0=time.time()
    try:
        cb(rt)
    except MemoryError: st='oom'
    except RematExceededError: st='thrashed'
    wall=time.time()-t0
    ev = rt.telemetry.trace.evict     # {time: [root_id,...]}
    # flatten to time-ordered stream
    stream=[]
    for tm in sorted(ev.keys()):
        for rid in ev[tm]:
            stream.append(rid)
    s=rt.telemetry.summary
    oh = (s['model_compute']+s['remat_compute'])/s['model_compute'] if st=='ok' and s['model_compute'] else None
    return dict(ratio=ratio, status=st, overhead=oh, wall=round(wall,1), stream=stream)

def analyze(tag, r):
    stream=r['stream']; n=len(stream)
    counts=Counter(stream); distinct=len(counts)
    top=counts.most_common(10)
    top_share = sum(c for _,c in top)/n*100 if n else 0
    # inter-eviction gap for the single most re-evicted storage
    hot = top[0][0] if top else None
    positions=[i for i,x in enumerate(stream) if x==hot]
    gaps=[positions[i+1]-positions[i] for i in range(len(positions)-1)]
    import statistics as st_
    print(f"\n===== {tag}  ratio={r['ratio']} status={r['status']} overhead={r['overhead']} wall={r['wall']}s =====")
    print(f"total evictions: {n}   distinct storages: {distinct}   evictions/storage: {n/distinct:.2f}")
    print(f"top-10 storages account for {top_share:.1f}% of all evictions")
    print(f"most re-evicted storage id {hot}: evicted {counts[hot]} times", end="")
    if gaps:
        print(f"; inter-eviction gap median={st_.median(gaps):.0f} mean={sum(gaps)/len(gaps):.1f} min={min(gaps)} max={max(gaps)}")
    else:
        print()
    print(f"top-10 (id:count): {top}")
    # cycle detection: within the tail, look at the sequence restricted to the top-6 ids
    top6=[i for i,_ in counts.most_common(6)]
    sub=[x for x in stream if x in set(top6)]
    print(f"top-6 victim subsequence length {len(sub)}; first 30: {sub[:30]}")
    return dict(tag=tag, ratio=r['ratio'], status=r['status'], overhead=r['overhead'],
                total_evictions=n, distinct=distinct, evics_per_storage=round(n/distinct,3),
                top10=top, top10_share_pct=round(top_share,1),
                hot_id=hot, hot_count=counts[hot],
                hot_gap_median=(st_.median(gaps) if gaps else None),
                hot_gap_mean=(round(sum(gaps)/len(gaps),1) if gaps else None))

def main():
    cb=build(); peak,bc=baseline(cb)
    print(f"LSTM peak={peak/1e6:.1f}MB base_compute={bc/1e6:.1f}ms")
    out={}
    for tag,ratio in [("FAST",0.265),("SLOW",0.264)]:
        r=run(ratio,peak,bc)
        out[tag]=analyze(tag,r)
    json.dump(out, open("results/lstm_victim_order.json","w"), indent=1, default=str)
    print("\nwrote results/lstm_victim_order.json")

t=threading.Thread(target=main); threading.stack_size(256*1024*1024)
t2=threading.Thread(target=main); t2.start(); t2.join()
