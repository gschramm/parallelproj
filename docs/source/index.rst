.. parallelproj documentation master file, created by
   sphinx-quickstart on Fri Dec 29 09:47:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

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

.. nbgallery::
    :caption: Examples
    :name: example-gallery

    notebooks/nontof_projections.pct.py

.. toctree::
    :hidden:
    :caption: Getting Started
    
    installation

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


Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
