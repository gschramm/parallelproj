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
        
           $ mamba install parallelproj

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge parallelproj

.. tip::

   parallelproj can not only project numpy CPU arrays, but also **cupy GPU arrays** (no memory transfer between host and GPU needed). To enable the latter, you have to install the cupy package as well.

To install parallelproj and cupy (optional and only if you have a CUDA GPU) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba

        .. code-block:: console
        
           $ mamba install parallelproj cupy<=13.4

    .. tab-item:: conda

        .. code-block:: console
        
           $ conda install -c conda-forge parallelproj cupy<=13.4

.. note::
   On conda-forge, CPU, CUDA 11 and CUDA 12 builds of `libparallelproj` are availalbe.
   conda / mamba should automatically install the correct version on your local CUDA system.
   Nevertheless, you can explicitly specify the CUDA version by adding the package `cuda-version=11.X` or `cuda-version=12.X` to the install command.

.. tip::

   parallelproj can also project **pytorch CPU and GPU tensors** 

To install parallelproj and pytorch (optional) from conda-forge, run

.. tab-set::

    .. tab-item:: mamba, pytorch with CUDA support

        .. code-block:: console
        
           $ mamba install parallelproj pytorch cupy<=13.4

    .. tab-item:: mamba, pytorch without CUDA support

        .. code-block:: console
        
           $ mamba install parallelproj pytorch

    .. tab-item:: conda pytorch with CUDA support

        .. code-block:: console
        
           $ conda install -c conda-forge parallelproj pytorch cupy<=13.4

    .. tab-item:: conda pytorch without CUDA support

        .. code-block:: console
        
           $ conda install -c conda-forge parallelproj pytorch

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
