"""Pipeline steps: select, clean, correct, impute, forecast and derive.

Every step follows a scikit-learn style ``fit``/``transform`` contract and has
a matching spec model. Lands in M1 (select, clean) through M4 (forecast).
"""
