"""The SignalSpec: a declarative, versioned JSON description of a pipeline.

Mirrors flex-pse's ``flexcore.config`` conventions: strict pydantic models,
``schema_version``, ``kind``-discriminated steps, migrations and a committed
JSON Schema. Lands in M1.
"""
