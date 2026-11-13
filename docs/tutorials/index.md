# Tutorials

The tutorials teach flex-signal step by step, from a first look at the data to multi-stream cleaning rules. Each
one is a [marimo](https://marimo.io) notebook that you can read here or run yourself. Each release adds the
tutorials for its new features.

Every tutorial uses flex-signal's synthetic plant data, with faults injected at known places. So each step can
be scored: how many faults it caught, how many it missed, and how much good data it removed by mistake.

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
