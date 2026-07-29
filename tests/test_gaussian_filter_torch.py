"""Regression tests for :class:`GaussianFilterOperator` with PyTorch.

Root cause guarded here: ``scipy.ndimage`` builds the Gaussian kernel in the
array namespace of ``sigma``.  A *tensor* ``sigma`` therefore yields a tensor
kernel that ``gaussian_filter1d`` reverses with a negative-step slice
(``weights[::-1]``), which PyTorch does not support
(https://github.com/pytorch/pytorch/issues/175240) -- this fires even for a
plain NumPy input array.  As defence-in-depth against scipy's array-API
delegation (scipy >= 1.16 / ``SCIPY_ARRAY_API``), the CPU input is also filtered
via a NumPy view.  parallelproj must therefore (a) coerce ``sigma`` to native
Python and (b) not rely on scipy's delegation.

These tests are torch specific and intentionally *not* parametrized over
``(xp, dev)`` (so they do not import ``config.pytestmark``).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

torch_available = importlib.util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(not torch_available, reason="torch not installed")


def test_gaussian_filter_operator_tensor_sigma() -> None:
    """A tensor-valued ``sigma`` must not break the filter (the DeepInv bug).

    Uses a plain NumPy input on purpose to show the failure is driven by the
    ``sigma`` type, not the input array type.
    """
    import numpy as np
    import torch
    from parallelproj.operators import GaussianFilterOperator

    shape = (8, 8, 4)
    x = np.ones(shape, dtype=np.float32)

    ref = GaussianFilterOperator(shape, sigma=1.5)(x)

    # scalar tensor sigma and per-axis tensor sigma
    y_scalar = GaussianFilterOperator(shape, sigma=torch.tensor(1.5))(x)
    y_vec = GaussianFilterOperator(shape, sigma=torch.tensor([1.5, 1.5, 1.5]))(x)
    assert np.allclose(np.asarray(y_scalar), ref, atol=1e-6)
    assert np.allclose(np.asarray(y_vec), ref, atol=1e-6)

    # and with a torch input array (result stays a torch CPU tensor)
    xt = torch.ones(shape, dtype=torch.float32)
    yt = GaussianFilterOperator(shape, sigma=torch.tensor(1.5))(xt)
    assert isinstance(yt, torch.Tensor) and yt.device.type == "cpu"
    assert np.allclose(np.asarray(yt), ref, atol=1e-6)


def test_gaussian_filter_operator_torch_cpu_float_sigma() -> None:
    """Basic backend check: torch CPU input + float sigma stays a torch tensor."""
    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter

    from parallelproj.operators import GaussianFilterOperator

    shape = (8, 8, 4)
    op = GaussianFilterOperator(shape, sigma=1.5)
    y = op(torch.ones(shape, dtype=torch.float32))

    assert isinstance(y, torch.Tensor) and y.device.type == "cpu"
    assert tuple(y.shape) == shape
    ref = gaussian_filter(np.ones(shape, dtype=np.float32), sigma=1.5)
    assert np.allclose(np.asarray(y), ref, atol=1e-6)


def test_gaussian_filter_operator_torch_cpu_scipy_array_api() -> None:
    """Force scipy's array-API delegation via ``SCIPY_ARRAY_API=1``.

    scipy reads ``SCIPY_ARRAY_API`` at import time, so this runs in a fresh
    interpreter with the env var set (keeping the rest of the suite on scipy's
    default behaviour).  Guards the CPU NumPy-view path against scipy computing
    natively in a torch input's namespace.
    """
    code = textwrap.dedent(
        """
        import torch
        from parallelproj.operators import GaussianFilterOperator

        shape = (8, 8, 4)
        op = GaussianFilterOperator(shape, sigma=1.5)
        x = torch.ones(shape, dtype=torch.float32)
        y = op(x)
        assert isinstance(y, torch.Tensor), type(y)
        assert tuple(y.shape) == shape
        assert bool(torch.isfinite(y).all())
        print("OK")
        """
    )
    env = dict(os.environ, SCIPY_ARRAY_API="1")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "OK" in proc.stdout
