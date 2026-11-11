# Explanation

These pages explain how flex-signal works and why it is designed the way it is.

```{toctree}
:maxdepth: 1

signal_frame_and_flags
spec_schema
steps_and_backends
time_and_units
```

Each future page arrives with the milestone that builds the feature it describes:

| Page | What it covers | Milestone |
|---|---|---|
| Conditions | The `fit_when`/`apply_when` expression language, with duration and flag helpers | M2 |
| Calibration against a reference | Correcting a stream from a trusted, intermittent reference | M2 |
| Imputation tiers | Interpolation, statistical and deep imputers, and when to use each | M2 |
| flex-pse integration | Aligning to a `TimeBlock` and handing data to flexparameterize | M3 |
| Forecasting and backtesting | Forecast issue times, exogenous inputs, blocked backtests | M4 |
| The orchestrator contract | The API rules flex-orchestrator's wrapper relies on | M6 |
| Acquirium integration | Reading streams, writing results, `PipelineApp` | M7 |
