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

> **Status: pre-alpha (M0, scaffold).** There is no user-facing functionality
> yet. See [PLAN.md](PLAN.md) for the architecture and milestones.

## Installation

```bash
git clone https://github.com/flex-pse/flex-signal.git
cd flex-signal
conda env create -f environment.yml   # or: pip install -e ".[dev,docs]"
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
