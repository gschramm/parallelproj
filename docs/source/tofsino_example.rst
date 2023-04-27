TOF sinogram projection example
-------------------------------

.. note::
   This example can be run using **numpy or cupy arrays** (if a CUDA GPU is available and the cupy package is installed).
   To have numpy/cupy agnostic code, we use the ``import ... as xp`` lines at the top.
   In case you want to use numpy (even when cupy is available), simply force ``import numpy as xp``.

.. literalinclude:: ../../examples/01_tof_sinogram_projections.py
   :language: python
   :linenos:
