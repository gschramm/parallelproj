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
- python (tested with  >= 3.7.6)
- numpy  (tested with  >= 1.18.1)
- matplotlib (tested with >= 3.2.1)
- numba (tested with >= 0.53)
- scipy (tested with >= 1.2)


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

To create a virtual conda environment and to install all python dependencies we need, execute
```
conda env create -f environment.yml
```

This will create a virtual conda environment called ```parallelproj``` which
can be activate via
```
conda activate parallelproj
```
If an NVidia GPU is present and you want to build the CUDA libraries, install the conda cuda-toolkit package from the nvidia channel and cupy from conda-forge
```
conda install -c nvidia cuda-toolkit
conda install -c conda-forge cupy
```

Compile the C/CUDA libraries as described above using the environment variable ```CONDA_PREFIX``` as ```CMAKE_INSTALL_PREFIX```
```
cd my_project_dir
mkdir build
cd build

(Linux): cmake -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX} ..
(Windows:) cmake -DCMAKE_INSTALL_PREFIX=%CONDA_PREFIX% ..

cmake --build . --target install --config release
```
After the libraries are compiled and installed, you have to define the environment variables ```PARALLELPROJ_C_LIB``` and ```PARALLELPROJ_CUDA_LIB``` (if the CUDA lib was compiled) pointing to the compiled libraries. The last call to cmake should print the installation paths of the libs to the command line.

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

In the examples sub directory you can find a few demo script that show how to use the projectors. Good examples to start with are ```fwd_back.py```, ```tof_pet_sino.py``` and ```tof_pet_lm.py``` which demonstrate a simple forward and back projection, a short sinogram and listmode OS-MLEM reconstruction on simulated data. You can run them via
```
python fwd_back.py
python tof_pet_sino.py
python tof_pet_lm.py
```
When imported, pyparallelproj will test whether a CUDA GPU is available or not and run all projections on the GPU using the CUDA libs if possible.

If you want to explicitely disable all visible GPUs (e.g. to test the OpenMP libraries) or you want to use a specific CUDA device, set the enviroment variable ```CUDA_VISIBLE_DEVICES```
