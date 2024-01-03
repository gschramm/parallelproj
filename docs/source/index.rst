.. parallelproj documentation master file, created by
   sphinx-quickstart on Fri Dec 29 09:47:08 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

parallelproj: a Python array API compatible library for fast tomographic projections
====================================================================================

**parallelproj** provides simple and fast
forward and back projectors for tomographic reconstruction
(non-TOF and TOF, sinogram ans listmode) in Python that are `python array API <https://data-apis.org/array-api/latest/>`_ 
compatible meaning that they can be used with a variety of python
array libraries (e.g. numpy, cupy, pytorch) and devices (CPU and CUDA GPUs).

**github repository** `<https://github.com/gschramm/parallelproj>`_

Features
--------

* **C/OpenMP** and **CUDA** implementations of **3D Joseph matched forward and back projectors**
* **Python array API compatible Python interface** (e.g. directly compatible with numpy, cupy, **pytorch**) 
* **non-TOF** and **TOF** versions of the projectors
* dedicated **sinogram** and **listmode** versions of the projectors
* available from `conda-forge <https://github.com/conda-forge/parallelproj-feedstock>`_

.. toctree::
    :hidden:
    
    installation
..
  .. nbgallery::
      :caption: Low-level examples
      :name: example-gallery
  
      notebooks/nontof_projections.pct.py
      notebooks/tof_sinogram_projections.pct.py
      notebooks/tof_listmode_projections.pct.py

.. toctree::
    :hidden:
    :caption: Examples

    auto_examples/index

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
