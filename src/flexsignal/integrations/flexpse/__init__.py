"""flex-pse adapter: align to a TimeBlock grid and hand data to flexparameterize.

flex-pse imports flex-signal, never the reverse at import time. This is the
only module allowed to touch flex-pse, and it does so lazily, inside
functions. Lands in M3.
"""
