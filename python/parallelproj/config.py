"""package configurations"""
import importlib
import GPUtil
import numpy as np
import numpy.typing as npt

from typing import Union
from types import ModuleType

# number of available CUDA devices
num_available_cuda_devices = len(GPUtil.getGPUs())
cuda_enabled = (num_available_cuda_devices > 0)
# check if cupy is available
cupy_enabled = (importlib.util.find_spec('cupy') is not None)

# define type for cupy or numpy array
if cuda_enabled:
    import cupy as cp
    import cupy.typing as cpt
    XPArray = Union[npt.NDArray, cpt.NDArray]
    XPFloat32Array = Union[npt.NDArray[np.float32], cpt.NDArray[np.float32]]
    XPShortArray = Union[npt.NDArray[np.int16], cpt.NDArray[np.int16]]
else:
    XPArray = npt.NDArray
    XPFloat32Array = npt.NDArray[np.float32]
    XPShortArray = npt.NDArray[np.int16]


def get_array_module(array) -> ModuleType:
    """return module of a cupy or numpy array

    Parameters
    ----------
    array : cupy or numpy array

    Returns
    -------
    cupy or numpy module
    """
    if cupy_enabled:
        return cp.get_array_module(array)
    else:
        return np