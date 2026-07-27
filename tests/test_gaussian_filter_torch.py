"""Regression tests for :class:`GaussianFilterOperator` with PyTorch CPU tensors.

Guards against a scipy array-API *delegation* issue: when scipy computes
natively on a torch tensor, ``scipy.ndimage.gaussian_filter1d`` reverses the
Gaussian kernel with a negative-step slice (``weights[::-1]``) that PyTorch does
not support (https://github.com/pytorch/pytorch/issues/175240).  parallelproj
must therefore filter on a NumPy view, so this never happens regardless of the
scipy version or the ``SCIPY_ARRAY_API`` env var.

These tests are torch-CPU specific and are intentionally *not* parametrized over
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


def test_gaussian_filter_operator_torch_cpu() -> None:
    """A torch CPU tensor stays a torch CPU tensor and matches the NumPy result."""
    import numpy as np
    import torch
    from scipy.ndimage import gaussian_filter

    from parallelproj.operators import GaussianFilterOperator

    shape = (8, 8, 4)
    op = GaussianFilterOperator(shape, sigma=1.5)

    x = torch.ones(shape, dtype=torch.float32)
    y = op(x)

    assert isinstance(y, torch.Tensor)
    assert y.device.type == "cpu"
    assert tuple(y.shape) == shape
    assert bool(torch.isfinite(y).all())

    ref = gaussian_filter(np.ones(shape, dtype=np.float32), sigma=1.5)
    assert np.allclose(np.asarray(y), ref, atol=1e-6)


def test_gaussian_filter_operator_torch_cpu_scipy_array_api() -> None:
    """Force scipy's array-API delegation via ``SCIPY_ARRAY_API=1``.

    scipy reads ``SCIPY_ARRAY_API`` at import time, so this runs in a fresh
    interpreter with the env var set (which also keeps the rest of the suite on
    scipy's default behavior).  Under delegation the pre-fix operator hit the
    unsupported ``weights[::-1]`` negative-step slice on modern scipy; with the
    NumPy-view fix it succeeds.
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
