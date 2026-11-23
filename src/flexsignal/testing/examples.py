"""Ready-made example steps, for tests and docs.

The spec tests check that :func:`one_of_every_step` covers every registered step
kind flex-signal provides (built in or an adapter), so adding one means adding an
example here.
"""

from __future__ import annotations

from flexsignal.spec import Quantity, StepSpec
from flexsignal.steps import clean, correct, forecast, impute, select

__all__ = ["one_of_every_step"]


def one_of_every_step() -> list[StepSpec]:
    """Return one configured instance of every built-in step kind.

    The options are chosen to exercise non-default values, so a round trip
    through JSON tests more than the defaults.
    """
    return [
        select.Align(freq="5min", method="linear", how="inner"),
        select.Resample(freq="15min", agg="median", min_count=2, name="to_15min"),
        select.Window(start="2025-01-06", end="2025-01-08T12:00:00Z"),
        select.Mask(keep_when="pump_on == 1", streams=["flow"]),
        select.Rename(mapping={"temperature": "temp"}),
        select.ConvertUnits("flow", to="L/s"),
        select.Scale("pressure", factor=100.0, units="kPa"),
        select.Combine(output="flow_best", streams=["reference_flow", "flow"]),
        select.Shift("pressure", by="-15min"),
        clean.Range("flow", min=0.0, max=Quantity(value=500, units="m^3/hr")),
        clean.RateOfChange("flow", max_rate=200, per="1h", reset_after=4),
        clean.Stuck("flow", min_duration="1h", tolerance=0.0, exclude_values=[0.0]),
        clean.Hampel(["flow", "pressure"], window="1h", n_sigma=6),
        clean.Tukey("pressure", k=3),
        clean.Expression("pressure", invalid_when="pressure < 0", flag="OUT_OF_RANGE"),
        clean.Filter("temperature", method="savgol", window="1h", polyorder=3),
        clean.Detotalize("pressure", per="1min", rollover=1e6, units="bar/min"),
        correct.Calibrate(
            target="flow",
            reference="reference_flow",
            fit_when="pump_on == 1 and held(pump_on == 1, '30min')",
            apply_when="not flagged(flow, 'STUCK')",
            model="sklearn:HuberRegressor",
            params={"epsilon": 1.5},
            mode="residual",
            features=["flow", "temperature"],
            refit={"rolling": "2D", "every": "12h", "min_rows": 20},
            output="flow_cal",
        ),
        impute.Interpolate("flow", max_gap="15min", method="pchip"),
        impute.Kalman("flow", level="local_linear_trend", seasonal=[{"period": "1D"}]),
        impute.Regression(
            "flow",
            features=["pressure"],
            lags=["5min"],
            fourier=[{"period": "1D", "harmonics": 2}],
            model="statsmodels:OLS",
            max_gap="6h",
        ),
        impute.KNN("flow", features=["pressure"], n_neighbors=3, weights="distance"),
        impute.Iterative("flow", features=["pressure"], max_iter=5),
        impute.Tiered(
            "flow",
            tiers=[
                {"max_gap": "15min", "method": "interpolate"},
                {
                    "max_gap": None,
                    "method": "regression",
                    "options": {"features": ["pressure"]},
                },
            ],
        ),
        forecast.Persistence("flow", horizon="1h", quantiles=[0.1, 0.9]),
        forecast.SeasonalNaive(
            "flow", horizon="6h", season="1D", issue_time="2025-01-08T00:00:00Z"
        ),
        forecast.Structural(
            "flow",
            horizon="6h",
            level="local_linear_trend",
            seasonal=[{"period": "1D", "harmonics": 2}],
            autoregressive=1,
            exog=["temperature"],
            quantiles=[0.05, 0.5, 0.95],
        ),
        forecast.Regression(
            "flow",
            horizon="1h",
            fourier=[{"period": "1D", "harmonics": 2}],
            lags=["5min", "1D"],
            past=["pressure"],
            past_lags=["1h"],
            exog=["temperature"],
            model="statsmodels:OLS",
            name="flow_ahead",
        ),
        # flex-signal's adapters: registered methods, built the same way.
        clean.PyodDetector("flow", model="ECOD", contamination=0.02),
        clean.RupturesChangepoint("flow", max_segment="1h"),
        clean.StumpyDiscord("flow", window="2h", k=8.0),
        impute.PypotsImpute("flow", model="BRITS", features=["pressure"], max_gap="6h"),
        forecast.PypotsForecast("flow", horizon="6h", model="TimesNet"),
        forecast.ChronosForecast("flow", horizon="1D", context="7D"),
        forecast.T0Forecast("flow", horizon="1D", context="7D"),
        forecast.NeuralforecastForecast("flow", horizon="1D", model="MLP"),
        forecast.DartsForecast("flow", horizon="1D", model="Theta"),
        forecast.SktimeForecast("flow", horizon="1D", model="AutoETS"),
    ]
