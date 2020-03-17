# parallelproj
code for parallel TOF and NONTOF projections

## dependencies
- cmake>=3.9 (3.9 version needed to detect CUDA correctly)
- recent c compiler
- CUDA (tested with 10.1.105)

for the examples:
- python (tested with 3.7.6)
- numpy  (tested with 1.18.1)

## build the project
```
cd my_project_dir
mkdir build
cmake ..
make 
make install
```

The CMakeLists.txt is configured to search for CUDA.
If CUDA is not present, compilation of the CUDA lib is skipped.

After a successfull build and install, the compiled libs
should appear in the ./lib directory.

If doxygen is present, the documentation is rendered in 
html and latex in the ./doc directory.

## run examples
All python examples should have the "-h" option to show the meaning of the command line options.

OpenMP CPU examples:
```
cd my_project_dir
cd examples

python openmp_nontof_sino_scaling.py --nv 1
python openmp_tof_sino_scaling.py --nv 1 --nrep 1

python openmp_nontof_lm_scaling.py --ne 1e6
python openmp_tof_lm_scaling.py --ne 1e6
```
CUDA examples:
```
cd my_project_dir
cd examples

python cuda_nontof_sino_scaling.py --nv 1
python cuda_tof_sino_scaling.py --nv 1 --nrep 1

python cuda_nontof_lm_scaling.py --ne 1e6
python cuda_tof_lm_scaling.py --ne 1e6
```