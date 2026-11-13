# flex-signal

Timeseries selection, cleaning, imputation and forecasting for the
[flex-pse](https://github.com/flex-pse/flex-pse) ecosystem.

flex-signal takes raw plant data and selects, cleans, corrects, imputes and
forecasts it. The result can go straight into flex-pse models,
`flexparameterize` surrogate fits, and
[flex-orchestrator](https://github.com/flex-pse/flex-orchestrator) studies.
It is a wrapper: the algorithms come from pandas, scipy, scikit-learn,
statsmodels, PyPOTS and other established libraries. flex-signal adds:

- a declarative, versioned pipeline spec in JSON;
- per-sample quality flags and provenance;
- adapters to the rest of the ecosystem.

> **Status: pre-alpha (M1).** Cleaning pipelines work: select, clean and
> interpolate steps, JSON specs, fitted-pipeline persistence and synthetic test
> data. The API may still change before 1.0. See [PLAN.md](PLAN.md) for the
> architecture and milestones.

New here? The [tutorials](tutorials/) walk through the API from a first look
at the data to multi-stream cleaning rules (`pip install -e ".[tutorials]"`,
then `marimo edit tutorials/01_first_look.py`). They're also on the
[docs site](https://flex-pse.github.io/flex-signal/tutorials/index.html).

## Quick look

```python
import flexsignal as fs
from flexsignal.testing import synthetic

raw = synthetic.plant(seed=0).with_spikes("flow", count=5).signal()
pipe = fs.Pipeline(
    [
        fs.clean.Range("flow", min=0, max=500),
        fs.clean.Hampel("flow", window="1h", n_sigma=6, min_scale=1.0),
        fs.impute.Interpolate("flow", max_gap="15min"),
    ],
    name="flow_cleanup",
)
result = fs.run(pipe.spec, raw)
result.signal.flag_counts()["flow"]   # what was removed, and why
fs.dump_spec(pipe.spec)               # the same pipeline as JSON
```

## Installation

```bash
git clone https://github.com/flex-pse/flex-signal.git
cd flex-signal
conda env create -f environment.yml   # or: pip install -e ".[dev,docs,tutorials]"
conda activate flex-signal
```

flex-signal needs Python 3.11 or newer.

## Development

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
pytest                                   # tests live next to the code, in src/
ruff check . && black --check . && lint-imports
sphinx-build -W --keep-going -b html docs docs/_build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Documentation

<https://flex-pse.github.io/flex-signal/>

## License

Apache 2.0. See [LICENSE](LICENSE).
