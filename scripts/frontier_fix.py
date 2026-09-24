import sys, threading, math, json
sys.setrecursionlimit(2000000)
from simrd.parse import parse_file
from simrd.runtime import RuntimeV2EagerOptimized
from simrd.heuristic import Heuristic
from simrd.heuristic.dtr import DTR
LOG="/home/claude/dtr-prototype/simrd/logs/resnet32-56-9000000000.0-2020-10-1-13-3-30-default.log"
def build():
    with open(LOG) as f: return parse_file(f, start=False).get_closure()
def peakc():
    rt=RuntimeV2EagerOptimized(math.inf, Heuristic(), stats=False, trace=False)
    build()(rt); return rt.telemetry.summary['max_memory']
def distinct_locked(rt):
    seen={}
    for t in rt.tensor_map.values():
        s=t.storage
        if s.material and s.ref_int>0:
            seen[id(s)]=s.size
    return len(seen), sum(seen.values())
class P(RuntimeV2EagerOptimized):
    def __init__(self,*a,**k):
        super().__init__(*a,**k); self.depth=0; self.cur=None; self.peak=(-1,None)
    def _materialize(self,t,rematerialize=True):
        self.depth+=1; self.cur=t
        try: return super()._materialize(t,rematerialize=rematerialize)
        finally: self.depth-=1
    def _free(self,size):
        # measure distinct locked at entry
        nlock,lbytes=distinct_locked(self)
        if lbytes>self.peak[0]:
            self.peak=(lbytes, dict(n_distinct_locked=nlock, locked_bytes=lbytes,
                                    depth=self.depth, tensor=(self.cur.name if self.cur else None),
                                    requested=size))
        return super()._free(size)
peak=peakc()
def run(r):
    rt=P(int(peak*r),DTR(),stats=False,trace=False,remat_limit=math.inf)
    st='ok'
    try: build()(rt)
    except MemoryError: st='oom'
    return st, rt, int(peak*r)
out={}
for r in (0.101,0.104,0.107):
    st,rt,b=run(r); pk=rt.peak[1]
    out[f"{r}"]=dict(status=st, budget=b, **pk)
    print(f"r={r} status={st} budget={b/1e6:.1f}MB  distinct_locked={pk['n_distinct_locked']} "
          f"locked={pk['locked_bytes']/1e6:.1f}MB depth={pk['depth']} tensor={pk['tensor']} req={pk['requested']/1e6:.1f}MB")
json.dump(out, open('results/frontier_fixed.json','w'), indent=1, default=str)
threading.stack_size(256*1024*1024)
t=threading.Thread(target=lambda: None); t.start(); t.join()
