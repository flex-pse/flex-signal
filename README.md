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

> **Status: alpha (0.1.0).** Pipelines select, clean, calibrate a stream
> against an intermittent reference, impute gaps by length and forecast, from
> Python or JSON specs, with backtests against baselines. Methods of your own,
> and adapters to PyPOTS, Chronos-2, pyod, darts, sktime and more, run the same
> way. Cleaned and forecast data goes into flex-pse models on their own time
> steps, and is prepared for fitting surrogates. The spec schema is frozen from 0.1.0: specs
> saved now keep loading in later releases
> ([stability](https://flex-pse.github.io/flex-signal/explanation/stability.html)).

New here? The [tutorials](https://github.com/flex-pse/flex-signal/tree/main/tutorials) walk through the API from a first look
at the data to calibration and imputer benchmarks (`pip install -e ".[tutorials]"`,
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
pip install flex-signal              # the core: pandas, scipy, scikit-learn, statsmodels
pip install "flex-signal[all]"       # plus every optional backend
```

flex-signal needs Python 3.11 or newer. The
[installation guide](https://flex-pse.github.io/flex-signal/getting_started/installation.html) lists the extras
and how to add flex-pse.

For development, from a clone:

```bash
git clone https://github.com/flex-pse/flex-signal.git
cd flex-signal
conda env create -f environment.yml   # or: pip install -e ".[dev,docs,tutorials]"
conda activate flex-signal
```

## Development

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
pytest                                   # tests live next to the code, in src/
ruff check . && black --check . && lint-imports
sphinx-build -W --keep-going -b html docs docs/_build
```

See [CONTRIBUTING.md](https://github.com/flex-pse/flex-signal/blob/main/CONTRIBUTING.md) before opening a pull request.

## Documentation

<https://flex-pse.github.io/flex-signal/>

## License

Apache 2.0. See [LICENSE](https://github.com/flex-pse/flex-signal/blob/main/LICENSE).
