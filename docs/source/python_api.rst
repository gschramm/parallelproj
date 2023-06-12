linear operators
----------------

.. autoclass:: parallelproj.LinearOperator

.. autoclass:: parallelproj.MatrixOperator

.. autoclass:: parallelproj.ElementwiseMultiplicationOperator

.. autoclass:: parallelproj.GaussianFilterOperator

.. autoclass:: parallelproj.CompositeLinearOperator

.. autoclass:: parallelproj.VstackOperator


high-level projection operators
-------------------------------

.. autoclass:: parallelproj.ParallelViewProjector2D

low-level non-TOF projector API
-------------------------------

.. autofunction:: parallelproj.joseph3d_fwd

.. autofunction:: parallelproj.joseph3d_back


low-level TOF projector API
---------------------------

.. autofunction:: parallelproj.joseph3d_fwd_tof_sino
.. autofunction:: parallelproj.joseph3d_back_tof_sino

low-level TOF listmode projector API
------------------------------------

.. autofunction:: parallelproj.joseph3d_fwd_tof_lm
.. autofunction:: parallelproj.joseph3d_back_tof_lm
