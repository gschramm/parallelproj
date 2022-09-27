# parallelproj

OpenMP and CUDA libraries for 3D Joseph non-TOF and TOF forward and back projectors.

This project provides OpenMP and CUDA implementations of 3D Joseph non-TOF and TOF forward and back projectors that can be e.g. used for image reconstruction. The input to the projectors are a list of start and end points for line of responses (LORs) such that they are very flexible and suitable for sinogram and listmode processing.

## Installation

### Installation from conda

The easiest and recommended way to install the parallelproj OpenMP and CUDA libs is to get them from our conda channel via

```
conda install -c gschramm -c conda-forge parallelproj
```

### Building from source

### Dependencies

- cmake>=3.16 (3.16 version needed to detect CUDA correctly), we recommend to use cmake>=3.23
- a recent c compiler with OpenMP support (tested with gcc 9.3 and msvc)
- git (to detect the version from git describe)
- cuda toolkit (optional, tested with >= 10)

**Notes**

- if cuda is not available on the build system, the build of the cuda library is skipped (only the C/OpenMP library is build)
- to install all dependencies for the build, you can use our conda
  [environment.yml](environment.yml)
- make sure to use `git clone` to get the sources, since we use `git describe` to deduce the cmake project version

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

### Setting CMAKE_CUDA_ARCHITECTURES

If you have CUDA available on your system (even if there is no physical CUDA GPU),
the default for `CMAKE_CUDA_ARCHITECTURES` depends on the cmake version you are using.

- **cmake version >= 3.23**: If you are using cmake >= 3.23, then by default `CMAKE_CUDA_ARCHITECTURES=all` which means that the code is build
  for all CUDA architectures.

- **3.16 <= cmake version < 3.23**: If you are using cmake < 3.23, then the default of `CMAKE_CUDA_ARCHITECTURES` is set to the architecture that is present on your system. **This means that if you are compiling on a system without physical CUDA GPU and using cmake < v3.23, you have to set it manually**, e.g. via `-DCMAKE_CUDA_ARCHITECTURES=75`.
