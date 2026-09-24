import sys, threading, math, json
sys.setrecursionlimit(2000000)
from simrd.parse import parse_file
from simrd.runtime import RuntimeV2EagerOptimized, RematExceededError
from simrd.heuristic import Heuristic
from simrd.heuristic.dtr import DTR

LOG="/home/claude/dtr-prototype/simrd/logs/resnet32-56-9000000000.0-2020-10-1-13-3-30-default.log"

def build():
    with open(LOG) as f:
        return parse_file(f, start=False).get_closure()

def peakc():
    rt=RuntimeV2EagerOptimized(math.inf, Heuristic(), stats=False, trace=False)
    build()(rt); s=rt.telemetry.summary
    return s['max_memory'], s['model_compute']

class Probe(RuntimeV2EagerOptimized):
    def __init__(self,*a,**k):
        super().__init__(*a,**k)
        self._cur=None; self.fail=None
        self.max_pinned=0; self.evinfail=0
    def _materialize(self, t, rematerialize=True):
        self._cur=t
        return super()._materialize(t, rematerialize=rematerialize)
    def _free(self, size):
        # snapshot at entry
        pool_bytes=sum(s.size for s in self.storage_pool)
        locked=self.memory_usage-pool_bytes
        if locked>self.max_pinned: self.max_pinned=locked
        try:
            return super()._free(size)
        except MemoryError:
            t=self._cur
            self.fail=dict(
                clock=self.clock,
                fail_tensor_id=(t.id if t else None),
                fail_tensor_name=(t.name if t else None),
                requested_size=size,
                budget=self.budget,
                resident_at_fail=self.memory_usage,   # pool empty => all locked
                pool_len_at_fail=len(self.storage_pool),
                pool_bytes_at_fail=sum(s.size for s in self.storage_pool),
                locked_plus_request=self.memory_usage+size,
                over_budget_by=self.memory_usage+size-self.budget,
            )
            raise

def run(ratio, peak, bc, remat_limit):
    rt=Probe(int(peak*ratio), DTR(), stats=False, trace=False, remat_limit=remat_limit)
    st='ok'
    try: build()(rt)
    except MemoryError: st='oom'
    except RematExceededError: st='thrashed'
    return st, rt

def main():
    peak,bc=peakc()
    print(f"peak={peak/1e6:.1f}MB base_compute={bc/1e6:.1f}ms\n")
    res={}
    # 0.104 with NO overhead cap (remat_limit=inf) -> rule out patch artifact
    st,rt=run(0.104, peak, bc, math.inf)
    print(f"[0.104, remat_limit=inf] status={st}")
    if rt.fail:
        f=rt.fail
        for k,v in f.items():
            vv=f"{v/1e6:.2f} MB" if k in('requested_size','budget','resident_at_fail','pool_bytes_at_fail','locked_plus_request','over_budget_by') else v
            print(f"    {k}: {vv}")
        res['fail_0.104']=f
    Tstar=rt.fail['fail_tensor_id'] if rt.fail else None
    # does the same tensor compute fine at 0.101 and 0.107? capture pinned peak
    for r in (0.101,0.107):
        st,rt=run(r, peak, bc, math.inf)
        print(f"\n[{r}, remat_limit=inf] status={st}  max_pinned_during_run={rt.max_pinned/1e6:.1f} MB  budget={int(peak*r)/1e6:.1f} MB")
        res[f'run_{r}']=dict(status=st, max_pinned_MB=round(rt.max_pinned/1e6,1), budget_MB=round(int(peak*r)/1e6,1))
    # also record 0.104 max_pinned & budget
    st,rt=run(0.104, peak, bc, math.inf)
    res['run_0.104']=dict(status=st, max_pinned_MB=round(rt.max_pinned/1e6,1), budget_MB=round(int(peak*0.104)/1e6,1))
    print(f"\n[0.104] max_pinned_during_run={rt.max_pinned/1e6:.1f} MB  budget={int(peak*0.104)/1e6:.1f} MB  fail_tensor={Tstar}")
    json.dump(res, open('results/oom_probe.json','w'), indent=1, default=str)
    print("\nwrote results/oom_probe.json")

threading.stack_size(256*1024*1024)
t=threading.Thread(target=main); t.start(); t.join()
