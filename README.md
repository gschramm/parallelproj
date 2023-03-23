# parallelproj

OpenMP and CUDA libraries for 3D Joseph non-TOF and TOF forward and back projectors.

This project provides OpenMP and CUDA implementations of 3D Joseph non-TOF and TOF forward and back projectors that can be e.g. used for image reconstruction. The input to the projectors are a list of start and end points for line of responses (LORs) such that they are very flexible and suitable for sinogram and listmode processing.

A few benchmarking tests of the projectors can be found in [this](https://arxiv.org/abs/2212.12519) preprint on arxiv.

## Installation

### (Option 1 - recommended) Installation from conda-forge

Precompiled parallelproj OpenMP and CUDA libraries and all their dependencies are available on [conda-forge](https://github.com/conda-forge/parallelproj-feedstock)
for Linux, Windows and MacOS. You can install the libraries via

```
conda install -c conda-forge parallelproj
```

_Remarks_:

- _As usual, we recommend to install the libs into a separate conda virtual enviornment._
- conda auto detects where cuda is available on your system
- if cuda is not available, only the OpenMP library will be installed
- currently the precompiled cuda library is not available for MacOS

### (Option 2) Building from source

### Dependencies

- cmake>=3.16 (3.16 version needed to detect CUDA correctly), we recommend to use cmake>=3.23
- a recent c compiler with OpenMP support (tested with gcc, msvc and clang)
- cuda toolkit (optional, tested with >= 10)

**Notes**

- If cuda is not available on the build system, the build of the cuda library is skipped (only the C/OpenMP library is build)
- If you are building on a recent version of Ubuntu with cuda, we recommend to use cuda >= 11.7. See [here](https://github.com/gschramm/parallelproj/issues/24) why.

### Building using cmake

We use CMake to auto generate a Makefile / Visual Studio project file which is used to compile the libraries. Make sure that cmake and the desired C compiler are on the PATH. The CMakeLists.txt is configured to search for CUDA. If CUDA is not present, compilation of the CUDA lib is skipped.

To build and install the libraries execute:

```
cd my_project_dir
mkdir build
cd build
cmake ..
cmake --build . --target install --config release
```

where `my_project_dir` is the directory that contains this file and the CMakeLists.txt file.
Note that for the default installation directory, you usually need admin priviledges.
To change the install directory, replace the 1st call to cmake by

```
cmake -DCMAKE_INSTALL_PREFIX=/foo/bar/myinstalldir ..
```

To build the documentation (doxygen required) run

```
cmake --build . --target docs
```

To run all unit tests execute:

```
ctest -VV
```


### Setting CMAKE_CUDA_ARCHITECTURES

If you have CUDA available on your system (even if there is no physical CUDA GPU),
the default for `CMAKE_CUDA_ARCHITECTURES` depends on the cmake version you are using.

- **cmake version >= 3.23**: If you are using cmake >= 3.23, then by default `CMAKE_CUDA_ARCHITECTURES=all` which means that the code is build
  for all CUDA architectures.

- **3.16 <= cmake version < 3.23**: If you are using cmake < 3.23, then the default of `CMAKE_CUDA_ARCHITECTURES` is set to the architecture that is present on your system. **This means that if you are compiling on a system without physical CUDA GPU and using cmake < v3.23, you have to set it manually**, e.g. via `-DCMAKE_CUDA_ARCHITECTURES=75`.

