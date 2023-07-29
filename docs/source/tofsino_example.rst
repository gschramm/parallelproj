Low-level TOF sinogram projection example
-----------------------------------------

.. note::
   parallelproj aims to be compatible with the **python array API** and 
   supports **different python array backends (numpy, cupy, pytorch)**.
   You can change the array backend in this example by commenting / uncommenting
   the import lines.

.. note::
   The example below shows how to do a simple TOF forward projection along
   a set of known lines of response with known start and end points in sinogram mode
   (using a set of fixed TOF bins along each LOR).



.. literalinclude:: ../../examples/01_tof_sinogram_projections.py
   :language: python
   :linenos:
