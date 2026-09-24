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
class P(RuntimeV2EagerOptimized):
    def __init__(self,*a,**k):
        super().__init__(*a,**k); self.depth=0; self.maxdepth=0; self.fail=None
    def _materialize(self,t,rematerialize=True):
        self.depth+=1; self.maxdepth=max(self.maxdepth,self.depth)
        try: return super()._materialize(t,rematerialize=rematerialize)
        finally: self.depth-=1
    def _free(self,size):
        try: return super()._free(size)
        except MemoryError:
            locked=[s for s in [] ]  # placeholder
            # count locked storages: those material with ref_int>0 (not in pool)
            allst=set()
            # gather from tensor_map storages
            n_locked=0; big=[]
            seen=set()
            for t in self.tensor_map.values():
                s=t.storage
                if id(s) in seen: continue
                seen.add(id(s))
                if s.material and s.ref_int>0:
                    n_locked+=1; big.append(s.size)
            big.sort(reverse=True)
            self.fail=dict(requested=size, budget=self.budget, resident=self.memory_usage,
                           n_locked=n_locked, top5_locked_MB=[round(b/1e6,1) for b in big[:5]],
                           remat_recursion_depth=self.depth)
            raise
peak=peakc()
def run(r):
    rt=P(int(peak*r),DTR(),stats=False,trace=False,remat_limit=math.inf)
    try: build()(rt)
    except MemoryError: pass
    return rt
def main():
    rt=run(0.104)
    print("OOM at 0.104:"); 
    print(json.dumps(rt.fail, indent=1))
    print("max remat recursion depth reached in run:", rt.maxdepth)
threading.stack_size(256*1024*1024)
t=threading.Thread(target=main); t.start(); t.join()
