Welcome to parallelproj's documentation!
========================================

**parallelproj** is a Python interface library for the
libparallelproj projector libraries written in C/OpenMP
and cuda.

.. note::
    The aim of **parallelproj** is to provide simple and fast
    forward and back projectors for tomographic reconstruction
    (non-TOF and TOF) in Python that are compatible with different
    array libraries (e.g. numpy, cupy, pytorch) and devices
    (CPU and CUDA GPUs).

.. toctree::
    :hidden:
    :caption: Getting Started
    
    installation

.. toctree::
    :hidden:
    :caption: Minimal Python Examples

    nontof_example
    tofsino_example
    toflm_example
    nontof_2d_projector
    nontof_3d_projector

.. toctree::
    :hidden:
    :caption: Python API

    python_api

.. toctree::
    :hidden:
    :caption: C/CUDA lib API

    libparallelproj_c
    libparallelproj_cuda
    cuda_kernels


