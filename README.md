# parallelproj
OpenMP and CUDA libraries and python bindings for 3D forward and back projectors
for PET reconstruction including time-of-flight (TOF) modeling

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

To build and install the libraries execute:
```
cd my_project_dir
mkdir build
cd build
cmake ..
cmake --build . --target install --config release
```
where ```my_project_dir``` is the directory that contains this file and the CMakeLists.txt file.
Note that for the default installation directory, you usually need admin priviledges.
To change the install directory, replace the 1st call to cmake by
```
cmake -DCMAKE_INSTALL_PREFIX=/foo/bar/myinstalldir ..
```
In case you want to use the libraries in combination with the python frontend package,
you can use the provided master python build script:
```
cd my_project_dir
python build_libs_and_wrappers.py
```
instead, which uses the CMAKE_INSTALL_PREFIX expected by the python package.

To build the documentation (doxygen required) run
```
cmake --build . --target docs
```

### (ii) Installation of the python package

The python package (and its dependencies) can be installed with pip. We recommend to install the package in a dedicated virtual environment if possible.

The pip installation can be done via:
```
cd my_project_dir
pip install .
```
which will install all python files and also copy the compiled libs to the directory specified by sys.prefix in the python sys module (e.g. the location where the virtual environment is located).

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
