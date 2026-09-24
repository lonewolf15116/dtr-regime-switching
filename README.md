# Reproducibility package — Deterministic Regime Switching and Feasibility Inversion in DTR

## Source
- Simulator: DTR public artefact, github.com/uwsampl/dtr-prototype
  simrd commit: eff53cc4804cc7d6246a6e5086861ce2b846f62b (2021-03-17)
- Traces (in simrd/logs/, from the artefact's logs.zip):
  - LSTM:     lstm-128-11000000000.0-2020-10-1-16-38-52-default.log
  - ResNet-32: resnet32-56-9000000000.0-2020-10-1-13-3-30-default.log

## Environment
- Reproduced identically on Python 3.11.15 (Linux) and Python 3.13 (Windows).
- Deps: attrs, dill, pathos, multiprocess, numpy (simrd's runtime deps).
  Do NOT install simrd/requirements.txt (2020-era pins); install the imports directly.
- PYTHONPATH must include the simrd package dir: dtr-prototype/simrd

## Configuration (identical across all runs unless noted)
- Runtime class: RuntimeV2EagerOptimized (required for DTR's region features;
  RuntimeV1 lacks the 'regions' feature and errors).
- Heuristics: DTR, AbESize, AbEStale, AbE (the last three from simrd/heuristic/ablation.py).
- Budget in bytes = int(unconstrained_peak_bytes * ratio).
- Overhead = (model_compute + remat_compute) / model_compute.
- Termination classes distinguished:
    * OOM       = MemoryError raised in RuntimeV2._free (evictable pool empty, allocation still doesn't fit)
    * thrashed  = RematExceededError (remat_compute exceeds remat_limit); NOT used for the OOM results
    * RecursionError = Python stack exhaustion during deep _materialize recursion (LSTM, very low budgets)
- Repeats: each point run as a fresh runtime; the ResNet OOM band was additionally
  reproduced in independent OS processes (one point per process).
- Instrumentation patches (do not change eviction decisions; they only widen limits / add logging):
    * sys.setrecursionlimit(1_000_000) + threading.stack_size(256 MB), run in a thread
      (needed for deep _materialize recursion; on Windows use 64 MB stack).
    * remat_limit set to inf for the OOM/frontier probes so OOM is not pre-empted by the overhead cap.
    * Trace/telemetry read-only hooks for the victim-order and frontier measurements.

## Commands
# Fine ResNet sweep (0.100–0.150, 0.001 steps, 3 repeats):
python stage1_probe.py <resnet_log> --ratios 0.100,0.101,...,0.150 --repeats 3 --overhead-limit 60 --out results/resnet_fine.json
# LSTM overhead switching (5 budgets, 3 repeats):
python stage1_probe.py <lstm_log> --ratios 0.268,0.266,0.265,0.264,0.262 --repeats 3 --overhead-limit 60 --out ...
# LSTM ablations: same, with --heuristic AbESize | AbEStale | AbE
# LSTM victim-order (fast vs slow, traced): python scripts/victim_order.py
# ResNet OOM allocation snapshot + frontier comparison: python scripts/oom_probe.py ; python scripts/frontier_fix.py

## Raw results included
- results/resnet_fine.json  (all 51 points, 3 repeats each, deterministic flags)
- results/lstm_victim_order.json  (fast/slow eviction-stream aggregates)
- results/oom_probe.json     (0.104 failure snapshot; 0.101/0.104/0.107 max pinned)
- results/frontier_fixed.json (distinct-locked counts, depth, tensor at peak frontier)
