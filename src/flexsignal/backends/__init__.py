"""Lazy adapters from a string key (``"sklearn:HuberRegressor"``) to a library.

Each adapter converts a SignalFrame into the shape its library expects and
back again, keeping the time index. Optional libraries are imported only when
their backend is used. Lands in M1 (registry) and grows through M5.
"""
