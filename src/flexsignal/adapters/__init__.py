"""flex-signal's own adapters to optional libraries, registered as methods.

Each module registers its methods through the public extension API
(:func:`flexsignal.register_cleaner` and its siblings), the same route a user's
methods take. Registering imports nothing heavy: the library is
imported when a step runs, and a missing library raises an error naming the
extra that installs it.
"""

from flexsignal.adapters import anomaly, darts, deep, foundation, pypots, sktime

__all__ = ["anomaly", "darts", "deep", "foundation", "pypots", "sktime"]
