import pytest

import numpy.array_api as nparr
import array_api_compat.numpy as np
import parallelproj

# generate list of array_api modules / device combinations to test
xp_dev_list = [(np, 'cpu')]

if np.__version__ >= '1.25':
    xp_dev_list.append((nparr, 'cpu'))

if parallelproj.cupy_enabled:
    import array_api_compat.cupy as cp
    xp_dev_list.append((cp, 'cuda'))

if parallelproj.torch_enabled:
    import array_api_compat.torch as torch
    xp_dev_list.append((torch, 'cpu'))

    if parallelproj.cuda_present:
        xp_dev_list.append((torch, 'cuda'))

pytestmark = pytest.mark.parametrize("xp,dev", xp_dev_list)
