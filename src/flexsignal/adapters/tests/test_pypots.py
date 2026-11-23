"""The PyPOTS adapters: windowing, PypotsImpute and PypotsForecast.

The windowing and option-checking tests need no PyPOTS; the training tests are
skipped when the ``pypots`` extra isn't installed.
"""

import importlib
import importlib.util
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import flexsignal as fs
from flexsignal.core import Flag, SignalBackendError, SignalConfigError, SignalDataError
from flexsignal.testing import synthetic

MODULE = "flexsignal.adapters.pypots"
NAMES = ("PypotsImpute", "PypotsForecast")
IMPUTE_MODELS = ("SAITS", "BRITS", "CSDI", "TimesNet")
FORECAST_MODELS = ("DLinear", "TimesNet", "Transformer", "TEFN", "CSDI")

needs_pypots = pytest.mark.skipif(
    importlib.util.find_spec("pypots") is None,
    reason="the pypots extra is not installed",
)


@pytest.fixture(scope="module", autouse=True)
def ap():
    """The adapter module; ``import flexsignal`` registers its methods."""
    return importlib.import_module(MODULE)


def _frame(n=10, columns=("a", "b"), freq="15min"):
    index = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC", name="time")
    data = np.arange(n * len(columns), dtype=float).reshape(n, len(columns))
    return pd.DataFrame(data, index=index, columns=list(columns))


def _plant(days=4, gap=True):
    """Flow and pressure every 15 min with a daily cycle, and a 5 h gap in flow."""
    case = synthetic.plant(periods=days * 96, freq="15min", seed=0)
    values = case.signal().values[["flow", "pressure"]].copy()
    if gap:
        values.iloc[100:120, 0] = np.nan
    return fs.SignalFrame.from_frame(
        values, units={"flow": "m^3/hr", "pressure": "bar"}
    )


# -- windowing (no PyPOTS needed) -----------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n", "window", "stride"),
    [
        (12, 4, None),  # whole windows
        (10, 4, None),  # a trailing partial window
        (10, 4, 3),  # overlapping, with a remainder
        (3, 5, None),  # shorter than one window: padded
        (10, 10, 2),  # exactly one window
    ],
)
def test_windows_round_trip_restores_index_and_values(ap, n, window, stride):
    df = _frame(n)
    df.iloc[1, 0] = np.nan
    df.iloc[-1, 1] = np.nan
    X = ap.to_windows(df, window, stride=stride)["X"]
    assert X.shape == (len(ap.window_starts(n, window, stride)), window, 2)
    back = ap.from_windows(X, df.index, df.columns, stride=stride)
    pd.testing.assert_frame_equal(back, df)


@pytest.mark.unit
def test_window_starts_cover_every_row(ap):
    assert ap.window_starts(10, 4).tolist() == [0, 4, 6]
    assert ap.window_starts(10, 4, 3).tolist() == [0, 3, 6]
    assert ap.window_starts(3, 5).tolist() == [0]
    with pytest.raises(SignalConfigError) as err:
        ap.window_starts(10, 4, 0)
    assert err.value.field == "stride"


@pytest.mark.unit
def test_overlapping_windows_are_averaged(ap):
    df = _frame(6, columns=("a",))
    X = ap.to_windows(df, 4, stride=2)["X"]  # starts 0, 2
    X[0] += 1.0  # the first window says one more
    back = ap.from_windows(X, df.index, ["a"], stride=2)
    expected = df["a"].to_numpy() + np.array([1, 1, 0.5, 0.5, 0, 0])
    np.testing.assert_allclose(back["a"].to_numpy(), expected)


@pytest.mark.unit
def test_windowing_explains_bad_input(ap):
    df = _frame(10)
    with pytest.raises(SignalDataError, match="regular time index"):
        ap.to_windows(df.iloc[[0, 1, 3, 4]], 2)
    X = ap.to_windows(df, 4)["X"]
    with pytest.raises(SignalDataError, match="windows"):
        ap.from_windows(X[:2], df.index, df.columns)
    with pytest.raises(SignalDataError, match="columns"):
        ap.from_windows(X, df.index, ["a"])


# -- registration and options (no PyPOTS needed) --------------------------------------


@pytest.mark.component
def test_importing_the_adapter_imports_no_heavy_library():
    code = (
        "import sys, flexsignal as fs, flexsignal.adapters.pypots; "
        "print(sorted(m for m in ('torch', 'pypots') if m in sys.modules)); "
        "print(fs.impute.PypotsImpute.__name__, fs.forecast.PypotsForecast.__name__)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split("\n")[:2] == ["[]", "PypotsImpute PypotsForecast"]


@pytest.mark.unit
def test_backend_keys_are_registered(ap):
    from flexsignal.backends import registered_backends

    keys = {b.key: b for b in registered_backends() if b.key.startswith("pypots:")}
    assert {f"pypots:imputation.{m}" for m in ap.IMPUTE_MODELS} <= set(keys)
    assert {f"pypots:forecasting.{m}" for m in ap.FORECAST_MODELS} <= set(keys)
    assert {info.extra for info in keys.values()} == {"pypots"}


@pytest.mark.unit
def test_option_descriptions_come_from_the_docstring():
    fields = fs.impute.PypotsImpute.model_fields
    assert "Length of the windows" in fields["window"].description
    assert fields["model"].default == "SAITS"
    assert fs.forecast.PypotsForecast.model_fields["model"].default == "DLinear"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "option", "value"),
    [
        ("PypotsImpute", "model", "pypots.imputation:SAITS"),
        ("PypotsImpute", "epochs", "many"),
        ("PypotsImpute", "features", "pressure"),
        ("PypotsImpute", "params", [1, 2]),
        ("PypotsForecast", "model", "os.system"),
        ("PypotsForecast", "samples", 1.5),
    ],
)
def test_bad_options_fail_on_load_with_the_field_named(kind, option, value):
    step = {"kind": kind, "streams": ["flow"], option: value}
    if kind == "PypotsForecast":
        step["horizon"] = "6h"
    spec = {"schema_version": "0.1.0", "requires": [MODULE], "steps": [step]}
    with pytest.raises(SignalConfigError) as err:
        fs.load_spec(spec)
    assert err.value.field == f"steps.0.{option}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "field"),
    [
        ({"d_model": 30, "n_heads": 4}, "d_model"),
        ({"epochs": 0}, "epochs"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"features": ["flow"]}, "features"),
        ({"features": ["nope"]}, "features"),
    ],
)
def test_invalid_values_are_explained_before_training(options, field):
    step = fs.impute.PypotsImpute("flow", **options)
    with pytest.raises(SignalConfigError) as err:
        fs.run(fs.Pipeline([step]).spec, _plant())
    assert err.value.field == field


@pytest.mark.unit
def test_data_problems_are_explained():
    with pytest.raises(SignalDataError, match="whole number"):
        fs.run(
            fs.Pipeline([fs.impute.PypotsImpute("flow", window="7min")]).spec, _plant()
        )
    with pytest.raises(SignalDataError, match="fewer than one"):
        fs.run(
            fs.Pipeline([fs.impute.PypotsImpute("flow", window="30D")]).spec, _plant()
        )
    step = fs.forecast.PypotsForecast("flow", horizon="6h", lookback="10D")
    with pytest.raises(SignalDataError, match="lookback"):
        fs.run(fs.Pipeline([step]).spec, _plant(gap=False))


@pytest.mark.unit
def test_exog_is_refused():
    sf = _plant(gap=False)
    values = sf.values
    values = values.reindex(
        pd.date_range(values.index[0], periods=len(values) + 24, freq="15min")
    )
    values["pressure"] = values["pressure"].ffill()
    sf = fs.SignalFrame.from_frame(values)
    step = fs.forecast.PypotsForecast("flow", horizon="6h", exog=["pressure"])
    with pytest.raises(SignalConfigError, match="no known-ahead") as err:
        fs.run(fs.Pipeline([step]).spec, sf)
    assert err.value.field == "exog"


@pytest.mark.unit
def test_a_missing_library_names_the_extra(monkeypatch):
    for name in ("pypots", "pypots.imputation", "pypots.forecasting"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(SignalBackendError, match=r"flex-signal\[pypots\]"):
        fs.run(fs.Pipeline([fs.impute.PypotsImpute("flow")]).spec, _plant())
    step = fs.forecast.PypotsForecast("flow", horizon="6h")
    with pytest.raises(SignalBackendError, match=r"flex-signal\[pypots\]"):
        fs.run(fs.Pipeline([step]).spec, _plant(gap=False))


# -- training (PyPOTS needed) ---------------------------------------------------------

TINY = {"epochs": 3, "d_model": 16, "n_layers": 1, "n_heads": 2}
FAST_CSDI = {"n_diffusion_steps": 10}  # CSDI is slow; fewer diffusion steps


def _impute_spec():
    return {
        "schema_version": "0.1.0",
        "name": "pypots",
        "requires": [MODULE],
        "steps": [
            {
                "kind": "PypotsImpute",
                "streams": ["flow"],
                "model": "SAITS",
                "window": "1D",
                "features": ["pressure"],
                "max_gap": "12h",
                **TINY,
            }
        ],
    }


def _forecast_spec(model="DLinear", quantiles=()):
    return {
        "schema_version": "0.1.0",
        "name": "pypots",
        "requires": [MODULE],
        "steps": [
            {
                "kind": "PypotsForecast",
                "streams": ["flow"],
                "horizon": "6h",
                "model": model,
                "lookback": "1D",
                "stride": "1h",
                "samples": 20,
                "quantiles": list(quantiles),
                "params": FAST_CSDI if model == "CSDI" else {},
                **TINY,
            }
        ],
    }


def _same_from_python_json_and_reload(spec, python_steps, sf, tmp_path):
    from_python = fs.run(fs.Pipeline(python_steps, name="pypots").spec, sf)
    from_json = fs.run(json.dumps(spec), sf)
    pd.testing.assert_frame_equal(from_python.data, from_json.data)
    pd.testing.assert_frame_equal(from_python.flags, from_json.flags)
    assert fs.load_spec(fs.dump_spec(from_json.fitted.spec)) == from_json.fitted.spec

    saved = from_json.fitted.save(tmp_path / "fitted")
    loaded = fs.FittedPipeline.load(saved, plugins=[MODULE])
    again = loaded.transform(sf)
    pd.testing.assert_frame_equal(again.values, from_json.signal.values)
    return from_json.signal


@needs_pypots
@pytest.mark.integration
def test_impute_runs_from_python_and_json_alike_and_reloads(tmp_path):
    sf = _plant(days=3)
    step = fs.impute.PypotsImpute(
        "flow", model="SAITS", window="1D", features=["pressure"], max_gap="12h", **TINY
    )
    out = _same_from_python_json_and_reload(_impute_spec(), [step], sf, tmp_path)
    filled = out.has_flag(Flag.IMPUTED_MODEL)["flow"]
    assert filled.sum() == 20
    assert out.values["flow"].notna().all()
    # Only the gap changes.
    np.testing.assert_array_equal(
        out.values["flow"][~filled], sf.values["flow"][~filled]
    )


@needs_pypots
@pytest.mark.integration
@pytest.mark.parametrize("model", IMPUTE_MODELS)
def test_every_imputation_model_fills_the_gap_and_repeats(model):
    sf = _plant(days=3)
    step = fs.impute.PypotsImpute(
        "flow",
        model=model,
        window="1D",
        stride="12h",
        samples=2,
        params=FAST_CSDI if model == "CSDI" else {},
        **TINY,
    )
    spec = fs.Pipeline([step]).spec
    first = fs.run(spec, sf).signal.values["flow"]
    second = fs.run(spec, sf).signal.values["flow"]
    pd.testing.assert_series_equal(first, second)
    assert np.isfinite(first).all()


@needs_pypots
@pytest.mark.integration
def test_unknown_params_are_refused_by_name():
    step = fs.impute.PypotsImpute("flow", params={"dropout": 0.1, "colour": 1}, **TINY)
    with pytest.raises(SignalConfigError, match="colour") as err:
        fs.run(fs.Pipeline([step]).spec, _plant())
    assert err.value.field == "params"
    step = fs.impute.PypotsImpute("flow", params={"epochs": 5}, **TINY)
    with pytest.raises(SignalConfigError, match="epochs"):
        fs.run(fs.Pipeline([step]).spec, _plant())


@needs_pypots
@pytest.mark.integration
def test_forecast_runs_from_python_and_json_alike_and_reloads(tmp_path):
    sf = _plant(gap=False)
    step = fs.forecast.PypotsForecast(
        "flow",
        horizon="6h",
        model="DLinear",
        lookback="1D",
        stride="1h",
        samples=20,
        **TINY,
    )
    out = _same_from_python_json_and_reload(_forecast_spec(), [step], sf, tmp_path)
    ahead = out.has_flag(Flag.FORECAST)["flow"]
    assert ahead.sum() == 24
    assert np.isfinite(out.values["flow"][ahead]).all()


@needs_pypots
@pytest.mark.integration
@pytest.mark.parametrize("model", FORECAST_MODELS)
def test_every_forecasting_model_forecasts_and_repeats(model):
    sf = _plant(days=3, gap=False)
    spec = _forecast_spec(model)
    first = fs.run(spec, sf).signal.values["flow"]
    second = fs.run(spec, sf).signal.values["flow"]
    pd.testing.assert_series_equal(first, second)
    assert np.isfinite(first.iloc[-24:]).all()


@needs_pypots
@pytest.mark.integration
def test_csdi_writes_quantiles_and_point_models_refuse_them():
    sf = _plant(days=3, gap=False)
    out = fs.run(_forecast_spec("CSDI", [0.1, 0.9]), sf).signal.values.iloc[-24:]
    assert (out["flow__q90"] > out["flow__q10"]).all()
    assert np.isfinite(out[["flow", "flow__q10", "flow__q90"]]).all().all()
    with pytest.raises(SignalConfigError, match="model='CSDI'") as err:
        fs.run(_forecast_spec("DLinear", [0.1, 0.9]), sf)
    assert err.value.field == "quantiles"


@needs_pypots
@pytest.mark.integration
def test_pypots_forecast_backtests():
    sf = _plant(days=4, gap=False)
    spec = fs.Pipeline(
        [
            fs.forecast.PypotsForecast(
                "flow", horizon="6h", lookback="1D", stride="2h", **TINY
            )
        ]
    ).spec
    result = fs.backtest(
        spec, sf, folds=2, compare=[fs.forecast.Persistence("flow", horizon="6h")]
    )
    summary = result.summary()
    assert set(summary["method"]) == {"PypotsForecast", "forecast_persistence"}


@needs_pypots
@pytest.mark.integration
def test_saits_benchmarks_against_tiered():
    """SAITS against interpolation + a seasonal Kalman smoother, on daily cycles."""
    sf = _plant(days=5, gap=False)
    imputers = {
        "tiered": fs.impute.Tiered(
            "flow",
            tiers=[
                {"max_gap": "30min", "method": "interpolate"},
                {
                    "max_gap": None,
                    "method": "kalman",
                    "options": {"seasonal": [{"period": "1D", "harmonics": 2}]},
                },
            ],
        ),
        "saits": fs.impute.PypotsImpute(
            "flow",
            model="SAITS",
            window="1D",
            stride="2h",
            features=["pressure"],
            epochs=60,
            batch_size=8,
            d_model=32,
            n_layers=2,
            n_heads=4,
        ),
    }
    results = fs.eval.benchmark_imputers(
        sf, imputers, streams=["flow"], gaps=("1h", "6h"), count=3
    )
    print(fs.eval.summarize(results, "rmse").to_string())
    assert set(results["imputer"]) == {"tiered", "saits"}
    assert np.isfinite(results["rmse"]).all()
    assert (results["coverage"] == 1.0).all()


@needs_pypots
@pytest.mark.integration
def test_training_is_quiet_and_leaves_pypots_logging_as_it_was(capfd):
    from pypots.utils.logging import logger

    level, handlers = logger.level, list(logger.handlers)
    fs.run(_forecast_spec(), _plant(days=3, gap=False))
    out, err = capfd.readouterr()
    assert "saving_path" not in out + err
    assert "Epoch" not in out + err
    assert (logger.level, logger.handlers) == (level, handlers)


@pytest.mark.unit
def test_the_model_lists_match_the_module(ap):
    assert ap.IMPUTE_MODELS == IMPUTE_MODELS
    assert ap.FORECAST_MODELS == FORECAST_MODELS
