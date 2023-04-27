Installation
============

.. _installation:

.. note::
    The **parallelproj** package consists of a C/OpenMP (libparallelproj_c), a CUDA (libparallelproj_cuda) projection library, and a python interface module (parallelproj). 
    We highly recommend to install parallelproj and pre-compiled version of the libs and the python interface from **conda-forge**.
    Pre-compiled libraries are available for all major operating system (with and without CUDA).

.. tip::

   You can get the **miniforge** conda install (minimal conda installer specific to conda-forge) `here <https://github.com/conda-forge/miniforge>`_.
   As usual, we recommend to install parallelproj into a separate **virtual environment**.

To install parallelproj (and the required compiled libraries) from conda-forge, run

.. code-block:: console

   $ mamba install parallelproj

or in case mamba is not available in your conda installation, run

.. code-block:: console

   $ conda install -c conda-forge parallelproj

.. tip::

   parallelproj can not only project numpy CPU arrays, but also **cupy GPU arrays** (no memory transfer between host and GOU needed). To enable the latter, you have to install the cupy package as well.

To install cupy (optional and only if you have a CUDA GPU) from conda-forge, run

.. code-block:: console

   $ mamba install cupy

or in case mamba is not available in your conda installation, run

.. code-block:: console

   $ conda install -c conda-forge cupy

.. note::
   In case you are interested in the compiled projection libraries, but not in the python interface, you can install the **libparallelproj** package from conda-forge.
