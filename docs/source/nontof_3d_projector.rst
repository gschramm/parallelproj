High-level 3D non-TOF projection example
----------------------------------------

.. note::
   parallelproj aims to be compatible with the **python array API** and 
   supports **different python array backends (numpy, cupy, pytorch)**.
   You can change the array backend in this example by commenting / uncommenting
   the import lines.

.. note::
   The example below shows how to do a non-TOF forward projection using a predefined
   3D parallelview projector.

.. literalinclude:: ../../examples/04_nontof_sinogram_3d_projector.py
   :language: python
   :linenos:
