# Deterministic Regime Switching and Feasibility Inversion in DTR

This repository contains the reproducibility package for an empirical
investigation of **Dynamic Tensor Rematerialization (DTR)** under constrained
memory.

The experiments identify deterministic regime changes in simulated DNN
training, including memory ranges in which increasing the available memory
can change a previously feasible execution into an out-of-memory execution.

The repository contains the experiment scripts, raw results, instrumentation,
and figures required to reproduce the reported observations.

> **Scope:** All experiments in this repository use the `simrd` simulator
> from the public DTR artefact. These results are simulator results, not
> measurements from a production DNN runtime.

## Source

- Simulator: DTR public artefact: `github.com/uwsampl/dtr-prototype`
- `simrd` commit:
  `eff53cc4804cc7d6246a6e5086861ce2b846f62b` (2021-03-17)
- Traces (in `simrd/logs/`, from the artefact's `logs.zip`):
  - LSTM:
    `lstm-128-11000000000.0-2020-10-1-16-38-52-default.log`
  - ResNet-32:
    `resnet32-56-9000000000.0-2020-10-1-13-3-30-default.log`

## Environment

- Reproduced identically on Python 3.11.15 (Linux) and Python 3.13 (Windows).
- Dependencies: `attrs`, `dill`, `pathos`, `multiprocess`, `numpy`
  (`simrd`'s runtime dependencies).
- Do **not** install `simrd/requirements.txt` (2020-era pins); install the
  imports directly.
- `PYTHONPATH` must include the `simrd` package directory:

```bash
dtr-prototype/simrd
