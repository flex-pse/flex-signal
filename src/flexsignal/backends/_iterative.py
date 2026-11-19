"""IterativeImputer is experimental in scikit-learn and needs an enabling import."""

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

__all__ = ["IterativeImputer"]
