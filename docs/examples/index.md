# Examples

Runnable example scripts. For a guided, step-by-step introduction, see the [tutorials](../tutorials/index.md).
Every example uses flex-signal's seeded synthetic generator, so it runs anywhere with no downloads. CI runs every example, and an example that needs an optional extra is skipped when that extra isn't
installed.

## Clean a week of synthetic plant data

Removes error codes, a frozen sensor, spikes and short gaps from a flow meter, and bad readings from a pressure
transmitter. It then checks the result against the faults it injected.

```{literalinclude} clean_synthetic_plant.py
:language: python
```
