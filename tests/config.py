import pytest

import array_api_strict as nparr
import array_api_compat.numpy as np
import parallelproj

# generate list of array_api modules / device combinations to test

xp_dev_list = []
xp_dev_list.append((np, "cpu"))
xp_dev_list.append((nparr, None))

if parallelproj.cupy_enabled:
    import array_api_compat.cupy as cp

    xp_dev_list.append((cp, "cuda"))

if parallelproj.torch_enabled:
    import array_api_compat.torch as torch

    xp_dev_list.append((torch, "cpu"))

    if parallelproj.cuda_present and parallelproj.cupy_enabled:
        xp_dev_list.append((torch, "cuda"))

pytestmark = pytest.mark.parametrize("xp,dev", xp_dev_list)
