Installation
============

.. note::
    The **parallelproj** package consists of a C/OpenMP (libparallelproj_c) projection library, 
    a CUDA (libparallelproj_cuda) projection library, and a python interface module (parallelproj). 
    We highly recommend to install parallelproj and pre-compiled versions of the libraries and the python interface from **conda-forge**.
    which are available for all major operating system (with and without CUDA).

.. tip::

   You can get the **miniforge** conda install (minimal conda installer specific to conda-forge) `here <https://github.com/conda-forge/miniforge>`_.
   As usual, we recommend to install parallelproj into a separate **virtual environment**.

To install parallelproj (and the required compiled libraries) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install libparallelproj parallelproj

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge libparallelproj parallelproj

.. tip::

   parallelproj can not only project numpy CPU arrays, but also **cupy GPU arrays** (no memory transfer between host and GPU needed). To enable the latter, you have to install the cupy package as well.

To install parallelproj and cupy (optional and only if you have a CUDA GPU) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install libparallelproj parallelproj cupy "numpy<2"

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge libparallelproj parallelproj cupy "numpy<2"

.. note::
   Support for numpy version 2 is not yet available in cupy and pytorch. Therefore, we have to restrict the numpy version for now (July 2024).
   In the future, this will change.


.. tip::

   parallelproj can also project **pytorch CPU and GPU tensors** 

To install parallelproj and pytorch (optional) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba, pytorch with CUDA support

        .. code-block:: console
        
           $ mamba install libparallelproj parallelproj pytorch cupy "numpy<2"

    .. tab-item:: mamba, pytorch without CUDA support

        .. code-block:: console
        
           $ mamba install libparallelproj parallelproj pytorch "numpy<2"

    .. tab-item:: conda pytorch with CUDA support

        .. code-block:: console
        
           $ conda install -c conda-forge libparallelproj parallelproj pytorch cupy "numpy<2"

    .. tab-item:: conda pytorch without CUDA support

        .. code-block:: console
        
           $ conda install -c conda-forge libparallelproj parallelproj pytorch "numpy<2"

Note that in case you want to use parallelproj with pytorch GPU tensors, cupy must be installed
next to pytorch as well, as shown in tabs above.

.. note::
   In case you are interested in the compiled projection libraries, but not in the python interface, you can install the **libparallelproj** package from conda-forge.

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install libparallelproj

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge libparallelproj
