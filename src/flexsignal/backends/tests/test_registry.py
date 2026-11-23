import numpy as np
import pytest

from flexsignal.backends import get_backend, register_backend, registered_backends
from flexsignal.core import SignalBackendError


@pytest.mark.unit
def test_registered_keys_resolve_and_import_paths_do_not():
    register_backend("test:norm", "numpy.linalg:norm")
    assert get_backend("test:norm") is np.linalg.norm
    keys = [b.key for b in registered_backends()]
    assert "scipy:PchipInterpolator" in keys
    # A key is never treated as an import path.
    with pytest.raises(SignalBackendError, match="unknown backend key"):
        get_backend("numpy.linalg:norm")


@pytest.mark.unit
def test_missing_extra_names_the_install_command():
    register_backend("fake:Model", "flexsignal_no_such_pkg.models:Model", extra="fake")
    with pytest.raises(SignalBackendError, match=r"pip install flex-signal\[fake\]"):
        get_backend("fake:Model")


@pytest.mark.unit
def test_unknown_key_and_conflicting_registration():
    with pytest.raises(SignalBackendError, match="unknown backend"):
        get_backend("nonsense")
    register_backend("test:missing", "numpy:no_such_function")
    with pytest.raises(SignalBackendError, match="no attribute"):
        get_backend("test:missing")
    with pytest.raises(SignalBackendError, match="already registered"):
        register_backend("numpy:interp", "numpy:polyfit")


@pytest.mark.unit
def test_builtin_keys_are_registered_in_one_module():
    from flexsignal.backends import builtin
    from flexsignal.backends.models import CURVE_FIT, LINEAR_MODELS

    keys = {info.key for info in registered_backends()}
    assert {*LINEAR_MODELS, CURVE_FIT, "scipy:PchipInterpolator"} <= keys
    assert builtin.__name__ == "flexsignal.backends.builtin"
    # A built-in key can't be silently redirected, even before any lookup.
    with pytest.raises(SignalBackendError, match="already registered"):
        register_backend("scipy:butter", "os:system")
