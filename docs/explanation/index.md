# Explanation

These pages explain how flex-signal works and why it is designed the way it is. Each page arrives with the
milestone that builds the feature it describes.

| Page | What it covers | Milestone |
|---|---|---|
| SignalFrame and quality flags | The data container, the per-sample flag bitmask, units and provenance | M1 |
| The pipeline spec | `SignalSpec`, step kinds, `schema_version`, migrations, input slots | M1 |
| Steps and backends | The fit/transform contract and how string keys reach libraries | M1 |
| Time and units | UTC inside flex-signal, local time at the flex-pse boundary, units metadata | M1 |
| Conditions | The `fit_when`/`apply_when` expression language | M2 |
| Calibration against a reference | Correcting a stream from a trusted, intermittent reference | M2 |
| Imputation tiers | Interpolation, statistical and deep imputers, and when to use each | M2 |
| flex-pse integration | Aligning to a `TimeBlock` and handing data to flexparameterize | M3 |
| Forecasting and backtesting | Forecast issue times, exogenous inputs, blocked backtests | M4 |
| The orchestrator contract | The API rules flex-orchestrator's wrapper relies on | M6 |
| Acquirium integration | Reading streams, writing results, `PipelineApp` | M7 |
