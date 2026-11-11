# Installation

flex-signal needs Python 3.11 or newer.

## From source

flex-signal is not on PyPI yet. Install it from a clone:

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

Without conda, use `pip install -e ".[dev,docs]"` instead. The pre-commit hooks expect the `flex-signal` conda
environment.

## Optional extras

Each heavy or specialised library is an optional extra, added in the milestone whose code uses it. They are not
available yet.

| Extra | Contents | Milestone |
|---|---|---|
| `eval` | pygrinder masking benchmarks | M2 |
| `flexpse` | The latest flex-pse release (v0.3.0 today) with `parameterize`, installed from its git tag until flex-pse is on PyPI | M3 |
| `ml` | skforecast, lightgbm | M4 |
| `nixtla` | statsforecast, mlforecast | M4 |
| `pypots` | PyPOTS deep imputation and forecasting | M5 |
| `deep` | neuralforecast | M5 |
| `foundation` | Chronos-2 zero-shot forecasting | M5 |
| `anomaly` | pyod, ruptures, stumpy | M5 |
| `darts`, `sktime` | Adapters to the darts and sktime model zoos | M5 |
| `units` | pint, pint-pandas | M5 |
| `acquirium` | Acquirium client | M7 |
