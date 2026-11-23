# Installation

flex-signal needs Python 3.11 or newer.

## From PyPI

```bash
pip install flex-signal
```

Add extras in brackets, e.g. `pip install "flex-signal[anomaly,units]"` (see below).

## From source

To work on flex-signal, or to use unreleased changes, install it from a clone:

```bash
git clone https://github.com/flex-pse/flex-signal.git
cd flex-signal
pip install -e .
```

The core install includes pandas, numpy, scipy, scikit-learn, statsmodels, pydantic, pyarrow and pandera. It does
not include torch, flex-pse, polars or Acquirium. `import flexsignal` never loads any of those.

## Development environment

The conda environment installs flex-signal in editable mode with the development and documentation tools:

```bash
conda env create -f environment.yml
conda activate flex-signal
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Without conda, use `pip install -e ".[dev,docs,tutorials]"` instead. The pre-commit hooks expect the `flex-signal` conda
environment.

## Optional extras

Each heavy or specialised library is an optional extra. Everything else, including the calibration models, the
Kalman, regression, k-NN and iterative imputers, the forecasters, the imputer benchmark and backtesting, uses the
core dependencies. A missing extra is reported by name when a step that needs it runs.

| Extra | Contents |
|---|---|
| `tutorials` | marimo (pinned) and matplotlib, to run the [tutorials](../tutorials/index.md) |
| `units` | pint, for `convert_units` and `range` bounds in other units |
| `anomaly` | pyod, ruptures and stumpy cleaners |
| `pypots` | PyPOTS deep imputation and forecasting, with pygrinder |
| `foundation` | Chronos-2 zero-shot forecasting |
| `t0` | t0-alpha zero-shot forecasting. Can't be installed alongside `pypots` |
| `deep` | neuralforecast |
| `darts`, `sktime` | Adapters to the darts and sktime model zoos |
| `all` | Every extra above except `tutorials` and `t0` |
| flex-pse | The latest flex-pse release (0.3.1 today), for `flexsignal.integrations.flexpse` and tutorial 07. Not on PyPI yet, so install it from its git tag (below); `environment.yml` already does |

Deep-learning extras pull in PyTorch. For a CPU-only install, put CPU torch in first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## flex-pse

`flexsignal.handoff` needs nothing extra. Fixing a flex-pse model's variables (`flexsignal.integrations.flexpse`),
the flex-pse how-to guides and tutorial 07 need flex-pse itself. It isn't on PyPI yet, so install the latest release
from its git tag, with HiGHS to solve:

```bash
pip install "flex-pse[parameterize,solvers] @ git+https://github.com/flex-pse/flex-pse.git@0.3.1"
```

`import flexsignal` never imports flex-pse, pyomo or idaes, so the rest of flex-signal works without it.
