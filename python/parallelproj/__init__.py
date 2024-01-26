from .backend import (
    cuda_present,
    cupy_enabled,
    torch_enabled,
    is_cuda_array,
    empty_cuda_cache,
)
from .backend import num_visible_cuda_devices
from .backend import joseph3d_fwd, joseph3d_back
from .backend import joseph3d_fwd_tof_sino, joseph3d_back_tof_sino
from .backend import joseph3d_fwd_tof_lm, joseph3d_back_tof_lm

from .operators import LinearOperator, MatrixOperator, ElementwiseMultiplicationOperator
from .operators import TOFNonTOFElementwiseMultiplicationOperator
from .operators import (
    GaussianFilterOperator,
    CompositeLinearOperator,
    VstackOperator,
    LinearOperatorSequence,
)
from .operators import FiniteForwardDifference

from .projectors import (
    ParallelViewProjector2D,
    ParallelViewProjector3D,
    RegularPolygonPETProjector,
)
from .projectors import ListmodePETProjector

from .pet_scanners import (
    RegularPolygonPETScannerModule,
    RegularPolygonPETScannerGeometry,
    DemoPETScannerGeometry,
)
from .pet_lors import SinogramSpatialAxisOrder, RegularPolygonPETLORDescriptor

from .tof import TOFParameters

__all__ = [
    "cuda_present",
    "cupy_enabled",
    "torch_enabled",
    "is_cuda_array",
    "empty_cuda_cache",
    "num_visible_cuda_devices",
    "joseph3d_fwd",
    "joseph3d_back",
    "joseph3d_fwd_tof_sino",
    "joseph3d_back_tof_sino",
    "joseph3d_fwd_tof_lm",
    "joseph3d_back_tof_lm",
    "LinearOperator",
    "MatrixOperator",
    "ElementwiseMultiplicationOperator",
    "TOFNonTOFElementwiseMultiplicationOperator",
    "GaussianFilterOperator",
    "CompositeLinearOperator",
    "VstackOperator",
    "LinearOperatorSequence",
    "FiniteForwardDifference",
    "ParallelViewProjector2D",
    "ParallelViewProjector3D",
    "RegularPolygonPETProjector",
    "ListmodePETProjector",
    "RegularPolygonPETScannerModule",
    "RegularPolygonPETScannerGeometry",
    "DemoPETScannerGeometry",
    "TOFParameters",
    "SinogramSpatialAxisOrder",
    "RegularPolygonPETLORDescriptor",
]

print(
    """
      _   _   _   _   _   _   _   _   _   _   _   _
     / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ / \ 
    ( P | A | R | A | L | L | E | L | P | R | O | J )
     \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/ \_/

    =================================================
      Please consider citing our publication
      ---------------------------------------------
      Georg Schramm and Kris Thielemans:
      "PARALLELPROJ—an open-source framework for
       fast calculation of projections in
       tomography"
      Front. Nucl. Med., 08 January 2024
      Sec. PET and SPECT, Vol 3
      https://doi.org/10.3389/fnume.2023.1324562
    =================================================

    """
)
