Installation
============

.. note::
    The **parallelproj** package consists of a C/OpenMP (libparallelproj_c), a CUDA (libparallelproj_cuda) projection library, and a python interface module (parallelproj). 
    We highly recommend to install parallelproj and pre-compiled versions of the libraries and the python interface from **conda-forge**.
    which are available for all major operating system (with and without CUDA).

.. tip::

   You can get the **miniforge** conda install (minimal conda installer specific to conda-forge) `here <https://github.com/conda-forge/miniforge>`_.
   As usual, we recommend to install parallelproj into a separate **virtual environment**.

To install parallelproj (and the required compiled libraries) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install parallelproj

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge parallelproj

.. tip::

   parallelproj can not only project numpy CPU arrays, but also **cupy GPU arrays** (no memory transfer between host and GPU needed). To enable the latter, you have to install the cupy package as well.

To install cupy (optional and only if you have a CUDA GPU) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install cupy

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge cupy

.. tip::

   parallelproj can also project **pytorch CPU and GPU tensors** 

To install pytorch from conda-forge, run

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install pytorch

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge pytorch


.. note::
   In case you are interested in the compiled projection libraries, but not in the python interface, you can install the **libparallelproj** package from conda-forge.

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install libparallelproj

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge libparallelproj

