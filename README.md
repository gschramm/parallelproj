# parallelproj
OpenMP and CUDA libraries and python bindings for 3D Joseph non-TOF and TOF forward and back projectors.

This project provided OpenMP and CUDA implementations of 3D Joseph non-TOF and TOF forward and back projectors that can be e.g. used for image reconstruction. The input to the projectors are a list of start and end points for line of responses (LORs) such that they are very flexible and suitable for sinogram and listmode processing.

On top of the projectors we also provide a few python bindings and some basic reconstruction examples.

## Dependencies

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
- numba (tested with v.1.2)


## Compilation of OpenMP (and CUDA libraries)

*If you want to use the libraries together with the python bindings, skip this section and continue with the next section.*

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
To build the documentation (doxygen required) run
```
cmake --build . --target docs
```

## Installation of the python bindings

We strongly recommend to use a virtual conda environment for the python bindings.

Download and install Miniconda from <https://docs.conda.io/en/latest/miniconda.html>.

Please use the ***Python 3.x*** installer and confirm that the installer
should run ```conda init``` at the end of the installation process.

To create a virtual conda environment and to install all python dependencies execute
```
conda create -n parallelproj "python>=3.7" "numpy>=1.18" "matplotlib>=3.2" "numba >=0.49" "scipy>=1.2" "cmake>=3.9"
```

Activate your conda environment
```
conda activate parallelproj
```
and compile the C/CUDA libraries as describe above using the environment variable ```CONDA_PREFIX``` as ```CMAKE_INSTALL_PREFIX```
```
cd my_project_dir
mkdir build
cd build
(Linux): cmake -DCMAKE_INSTALL_DIR=${CONDA_PREFIX} ..
(Windows:) cmake -DCMAKE_INSTALL_DIR=%CONDA_PREFIX% ..
cmake --build . --target install --config release
```
This makes sure that the compiled libraries are installed in the correct place and will be found by the python bindings
(using find_library from ctypes.util)

Finally, add the package directory to your ```PYTHONPATH``` environment variable.

## Test the installation and run examples

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
