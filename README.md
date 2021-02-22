# parallelproj
OpenMP and CUDA libraries and python bindings for 3D forward and back projectors

## 0. Dependencies

For the OpenMP library (CPU version):
- cmake>=3.9 (3.9 version needed to detect CUDA correctly)
- a recent c compiler with OpenMP support (tested with gcc 9.3 and msvc)

For the CUDA library (optional):
- CUDA (tested with 10.1.105)

For the python bindings:
- python (tested with v.3.7.6)
- numpy  (tested with v.1.18.1)
- matplotlib (tested with v.3.2.1)
- numba (tested with v.0.49)

## 1. Installation

The installation consists of two parts:
1. compilation of the OpenMP (and CUDA) projector libraries
2. Installation of the python package containing the python bindings and examples

### (i) Compilation of OpenMP (and CUDA libraries)

We use CMake to auto generate a Makefile / Visual Studio project file which is used to compile the libraries. Make sure that cmake and the desired C compiler are on the PATH. The CMakeLists.txt is configured to search for CUDA. If CUDA is not present, compilation of the CUDA lib is skipped.

To build all libraries using cmake and install them in the directory expected by the python package, execute 

```
cd my_project_dir
python build_c_libs.py
```

where ```my_project_dir``` is the directory that contains this file and the CMakeLists.txt file.
After a successful build and install, the compiled libs should appear in the **my_project_dir/pyparallelproj/lib** directory. The linux libs should be called libparallelproj_c.so (and libparallelproj_cuda.so) and the Windows libs should be called parallelproj_c.dll (and parallelproj_cuda.dll).

If doxygen is present, the documentation of the C and CUDA sources is rendered in html in the ```my_project_dir/doc``` directory.
If the cmake variable ```PARALLELPROJ_INSTALL_DOCS``` is set to TRUE, the documentation is installed into```PARALLELPROJ_DOC_DIR```.

(note) The step above described how to use the provided build script that builds and installs
the libraries in the directory expected for the python package.
In case you do only want to build the C/CUDA libraries and you are not interested in the python
bindings, you can of course use / call cmake yourself.
To display the commands that are called by the build script, you can use ```python build_libs_for_python.py --dry``` and probably modify / skip ```CMAKE_INSTALL_PREFIX```.


### (ii) Installation of the python package

The python package (and its dependencies) can be installed with pip. We recommend to install the package in a dedicated virtual environment if possible.

The pip installation can be done via:
```
cd my_project_dir
pip install .
```
which will install all python files and also copy the compiled libs to the subfolder lib in the directory specified by sys.prefix in the python sys module (e.g. the location where the virtual environment is located).

(note) instead of installing the python package, you can also add the package directory to your
```PYTHONPATH``` environment variable.

## 3. Test the installation and run examples

To test whether the python package was installed correctly run the following in python.
```python
import pyparallelproj as ppp
``` 

To test whether the compiled OpenMP lib is installed correctly run
```python
import pyparallelproj as ppp
print(ppp.config.lib_parallelproj_c)
``` 

If the CUDA lib was compiled, test the installation via
```python
import pyparallelproj as ppp
print(ppp.config.lib_parallelproj_cuda)
``` 

In the examples sub directory you can find a few demo script that show how to use the projectors. Good examples to start with are ```fwd_back.py```, ```tof_pet_sino.py``` and ```tof_pet_lm.py``` which demonstrate a simple forward and back projection, a short sinogram and listmode OS-MLEM reconstruction on simulated data. To run them without a a GPU and CUDA, execute:

```
python fwd_back.py
python tof_pet_sino.py
python tof_pet_lm.py
```
To use the CUDA GPU projector on a single GPU run
```
python fwd_back.py --ngpus 1
python tof_pet_sino.py --ngpus 1
python tof_pet_lm.py --ngpus 1
```
The "ngpus" option specifies on many GPUs should be used. If set to -1, CUDA will auto determine the number of available inter-connected GPUs. The default value 0 means that the OpenMP-based CPU projectors are used.
