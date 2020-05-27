# parallelproj
code for parallel TOF and NONTOF projections

## dependencies
- cmake>=3.9 (3.9 version needed to detect CUDA correctly)
- recent c compiler
- CUDA (tested with 10.1.105)

for the examples:
- python (tested with 3.7.6)
- numpy  (tested with 1.18.1)

## build the C / CUDA libraries

### Linux
```
cd my_project_dir
mkdir build
cmake ..
make 
make install
```

### Windows (using MSVC)
```
cd my_project_dir
mkdir build
cmake -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE ..
cmake --build . --target INSTALL --config RELEASE
```

The CMakeLists.txt is configured to search for CUDA.
If CUDA is not present, compilation of the CUDA lib is skipped.

After a successfull build and install, the compiled libs
should appear in the ./lib directory.

If doxygen is present, the documentation is rendered in 
html and latex in the ./doc directory.

## run examples
