# Tutorials

The tutorials teach flex-signal step by step, from a first look at the data to calibration, imputation, forecasting, handing data to flex-pse models, methods of your own, and running specs from flex-orchestrator. Each
one is a [marimo](https://marimo.io) notebook that you can read here or run yourself.

Every tutorial uses flex-signal's synthetic data, with faults injected at known places and the true values
known. So each step can be scored: how many faults it caught, how many it missed, how much good data it removed
by mistake, and how close a corrected or filled value is to the truth.

<table>
  <thead>
    <tr><th>#</th><th>Tutorial</th><th>What you learn</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><a href="01_first_look.html">A first look at plant data</a></td>
      <td><code>SignalFrame</code>, units, quality flags, bringing your own data, and the synthetic test data</td>
    </tr>
    <tr>
      <td>2</td>
      <td><a href="02_cleaning_one_stream.html">Cleaning one stream</a></td>
      <td>Range, stuck-sensor, rate-of-change and Hampel checks, scored against the injected faults;
        filling short gaps; why step order matters</td>
    </tr>
    <tr>
      <td>3</td>
      <td><a href="03_pipelines_and_specs.html">Pipelines, specs and fitted state</a></td>
      <td>JSON specs, spec hashes, provenance, validation errors, fitting a step on training data, saving and
        loading a fitted pipeline</td>
    </tr>
    <tr>
      <td>4</td>
      <td><a href="04_several_streams.html">Several streams</a></td>
      <td>Named inputs, aligning sources with different clocks, finding and removing a delay, masking
        operating periods, rules over several streams, combining meters</td>
    </tr>
    <tr>
      <td>5</td>
      <td><a href="05_calibrating_against_a_reference.html">Calibrating against a reference</a></td>
      <td>Correcting a miscalibrated meter A from an intermittent, sometimes-wrong reference B:
        <code>fit_when</code>, preferring the reference, following drift with a rolling refit</td>
    </tr>
    <tr>
      <td>6</td>
      <td><a href="06_comparing_imputers.html">Comparing imputers by gap length</a></td>
      <td>The masking benchmark, why the best imputer depends on the gap length, and building a tiered imputer
        from the results</td>
    </tr>
    <tr>
      <td>7</td>
      <td><a href="07_raw_data_to_a_flexpse_model.html">From raw data to a flex-pse model</a></td>
      <td>Lining cleaned data up with a model's time steps in the plant's time zone, fixing a model's variables
        and solving it, and preparing fit data for a flex-pse regressor (needs flex-pse)</td>
    </tr>
    <tr>
      <td>8</td>
      <td><a href="08_forecasting_and_backtesting.html">Forecasting and backtesting</a></td>
      <td>Baselines, a structural model and a regression on a plant with daily and weekly cycles; intervals; a
        backtest that can't peek at the future; fixing tomorrow's forecast into a flex-pse model (needs
        flex-pse)</td>
    </tr>
    <tr>
      <td>9</td>
      <td><a href="09_deep_imputation_and_forecasting.html">Deep imputation and forecasting</a></td>
      <td>SAITS (PyPOTS) against the tiered imputer; zero-shot Chronos-2 and a PyPOTS network against the core
        forecasters in a backtest; licences (needs the [pypots] and [foundation] extras)</td>
    </tr>
    <tr>
      <td>10</td>
      <td><a href="10_your_own_methods.html">Your own methods</a></td>
      <td>Register a cleaner, an imputer and a model; score them against injected faults and built-in methods;
        run them from a JSON spec; save and reload; what is refused; packaging them as a plugin</td>
    </tr>
    <tr>
      <td>11</td>
      <td><a href="11_specs_as_orchestrator_studies.html">A cleaning spec as a flex-orchestrator study</a></td>
      <td>What an orchestrator wrapper relies on: named input slots, spec hashes, a params model that embeds
        the spec and its JSON Schema, <code>RunResult</code>, storing a fitted pipeline as a payload, errors
        with field paths, progress, and spec versions</td>
    </tr>
    <tr>
      <td>12</td>
      <td><a href="12_scheduling_a_biogas_digester.html">Scheduling a biogas digester</a></td>
      <td>From raw digester data to a scheduled day: cleaning and 15-minute steps, calibrating a gas meter
        against lab checks, an ARIMA surrogate from <code>fit_data</code>, a feed forecast, and a flex-pse
        feed-tank-digester-CHP model under a time-of-use tariff, baseline against schedule (needs flex-pse with
        the ARIMA surrogate: flex-pse <code>main</code>, the next release after 0.3.1)</td>
    </tr>
  </tbody>
</table>

The pages here are static: you can read the code and the results, but the sliders and drop-downs don't
respond. To use them, run the tutorials yourself.

## Running the tutorials yourself

From a clone of the repository:

```bash
pip install -e ".[tutorials]"
marimo edit tutorials/02_cleaning_one_stream.py   # edit and run, with live controls
marimo run tutorials/02_cleaning_one_stream.py    # the same, as an app with the code hidden
python tutorials/02_cleaning_one_stream.py        # run it as a script, as CI does
```

marimo notebooks are plain Python files. A cell reruns automatically when anything it depends on changes, so
what you see always matches the code.
