from .backend import (
    cuda_present,
    cupy_enabled,
    torch_enabled,
    is_cuda_array,
    to_numpy_array,
    empty_cuda_cache,
    num_visible_cuda_devices,
    count_event_multiplicity,
    joseph3d_fwd,
    joseph3d_back,
    joseph3d_fwd_tof_sino,
    joseph3d_back_tof_sino,
    joseph3d_fwd_tof_lm,
    joseph3d_back_tof_lm,
    lib_parallelproj_c_fname,
    lib_parallelproj_cuda_fname,
    cuda_kernel_file,
)

from .operators import (
    LinearOperator,
    MatrixOperator,
    ElementwiseMultiplicationOperator,
    TOFNonTOFElementwiseMultiplicationOperator,
    GaussianFilterOperator,
    CompositeLinearOperator,
    VstackOperator,
    LinearOperatorSequence,
    FiniteForwardDifference,
)

from .projectors import (
    ParallelViewProjector2D,
    ParallelViewProjector3D,
    RegularPolygonPETProjector,
    ListmodePETProjector,
    EqualBlockPETProjector,
)

from .pet_scanners import (
    RegularPolygonPETScannerModule,
    RegularPolygonPETScannerGeometry,
    DemoPETScannerGeometry,
    BlockPETScannerModule,
    ModularizedPETScannerGeometry,
)
from .pet_lors import (
    SinogramSpatialAxisOrder,
    RegularPolygonPETLORDescriptor,
    EqualBlockPETLORDescriptor,
)

from .tof import TOFParameters

__all__ = [
    "count_event_multiplicity",
    "cuda_present",
    "cupy_enabled",
    "torch_enabled",
    "is_cuda_array",
    "to_numpy_array",
    "empty_cuda_cache",
    "num_visible_cuda_devices",
    "joseph3d_fwd",
    "joseph3d_back",
    "joseph3d_fwd_tof_sino",
    "joseph3d_back_tof_sino",
    "joseph3d_fwd_tof_lm",
    "joseph3d_back_tof_lm",
    "lib_parallelproj_c_fname",
    "lib_parallelproj_cuda_fname",
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
    "EqualBlockPETProjector",
    "ListmodePETProjector",
    "RegularPolygonPETScannerModule",
    "RegularPolygonPETScannerGeometry",
    "DemoPETScannerGeometry",
    "BlockPETScannerModule",
    "ModularizedPETScannerGeometry",
    "TOFParameters",
    "SinogramSpatialAxisOrder",
    "RegularPolygonPETLORDescriptor",
    "EqualBlockPETLORDescriptor",
]

print(
    f"""
          -  -  -  -  -  -  -  -   -  -  -  -
          P  A  R  A  L  L  E  L | P  R  O  J
          -  -  -  -  -  -  -  -   -  -  -  -

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
 
    parallelproj C    lib         ..: {lib_parallelproj_c_fname}
    parallelproj CUDA lib         ..: {lib_parallelproj_cuda_fname}
    parallelproj CUDA kernel file ..: {cuda_kernel_file}
    parallelproj CUDA present     ..: {cuda_present}
    parallelproj cupy enabled     ..: {cupy_enabled}
    """
)
