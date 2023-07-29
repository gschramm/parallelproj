Low-level TOF listmode projection example
-----------------------------------------

.. note::
   parallelproj aims to be compatible with the **python array API** and 
   supports **different python array backends (numpy, cupy, pytorch)**.
   You can change the array backend in this example by commenting / uncommenting
   the import lines.

.. note::
   The example below shows how to do a simple TOF forward projection along
   a set of known lines of response with known start and end points in listmode.

.. literalinclude:: ../../examples/02_tof_listmode_projections.py
   :language: python
   :linenos:
