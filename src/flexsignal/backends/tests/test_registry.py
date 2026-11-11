import numpy as np
import pytest

from flexsignal.backends import get_backend, register_backend, registered_backends
from flexsignal.core import SignalBackendError


@pytest.mark.unit
def test_registered_and_path_keys_resolve():
    assert get_backend("numpy:interp") is np.interp
    assert get_backend("numpy.linalg:norm") is np.linalg.norm
    keys = [b.key for b in registered_backends()]
    assert "scipy:PchipInterpolator" in keys


@pytest.mark.unit
def test_missing_extra_names_the_install_command():
    register_backend("fake:Model", "flexsignal_no_such_pkg.models:Model", extra="fake")
    with pytest.raises(SignalBackendError, match=r"pip install flex-signal\[fake\]"):
        get_backend("fake:Model")


@pytest.mark.unit
def test_unknown_key_and_conflicting_registration():
    with pytest.raises(SignalBackendError, match="unknown backend"):
        get_backend("nonsense")
    with pytest.raises(SignalBackendError, match="no attribute"):
        get_backend("numpy:no_such_function")
    with pytest.raises(SignalBackendError, match="already registered"):
        register_backend("numpy:interp", "numpy:polyfit")
