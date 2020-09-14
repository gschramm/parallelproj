# parallelproj
OpenMP and CUDA libraries and python bindings for 3D forward and back projectors

## 0. Dependencies

For the OpenMP library (CPU version):
- cmake>=3.9 (3.9 version needed to detect CUDA correctly)
- a recent c compiler (tested with gcc and msvc)

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

On Linux run
```
cd my_project_dir
mkdir build
cd build
cmake ..
make 
make install
```

On Windows run
```
cd my_project_dir
mkdir build
cd build
cmake -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ..
cmake --build . --target INSTALL --config RELEASE
```
where ```my_project_dir``` is the directory that contains this file and the CMakeLists.txt file.
After a successful build and install, the compiled libs should appear in the **my_project_dir/lib** directory. The linux libs should be called libparallelproj.so (and libparallelproj_cuda.so) and the Windows libs should be called parallelproj.dll (and parallelproj_cuda.dll).



If doxygen is present, the documentation of the C and CUDA sources is rendered in html and latex in the ./doc directory.

### (ii) Installation of the python package

The python package (and its dependencies) can be installed with pip. We recommend to install the package in a dedicated virtual environment if possible.

The pip installation can be done via:
```
cd my_project_dir
pip install .
```
which will install all python files and also copy the compiled libs to the subfolder lib in the directory specified by sys.prefix in the python sys module (e.g. the location where the virtual environment is located).

## 3. Test the installation and run examples

To test whether the python package was installed correctly run the following in python.
```python
import pyparallelproj as ppp
``` 

To test whether the compiled OpenMP lib is installed correctly run
```python
import pyparallelproj as ppp
print(ppp.wrapper.lib_parallelproj) 
``` 

If the CUDA lib was compiled, test the installation via
```python
import pyparallelproj as ppp
print(ppp.wrapper.lib_parallelproj_cuda) 
``` 

In the examples sub directory you can find a few demo script that show how to use the projectors. Two good examples to start with are the osem.py and osem_lm.py which demonstrate a short sinogram and listmode OS-MLEM reconstruction on simulated data. To run them without a a GPU and CUDA, execute:

```
python osem.py
python osem_lm.py
```
To use the CUDA GPU projector on a single GPU run
```
python osem.py --ngpus 1
python osem_lm.py --ngpus 1
```
The "ngpus" option specifies on many GPUs should be used. If set to -1, CUDA will auto determine the number of available inter-connected GPUs. The default value 0 means that the OpenMP-based CPU projectors are used.



 


