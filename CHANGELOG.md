## 1.3.2 (April 18, 2023)
- chore: updated package.json, updated CHANGELOG.md, bumped 1.3.1 -> 1.3.2
- rename test folder
- lower absolute tolerance for forward TOF tests - (otherwise windows builds might fail)
- dummy commit

## 1.3.1 (April 17, 2023)
- chore: updated package.json, updated CHANGELOG.md, bumped 1.3.0 -> 1.3.1
- add num_visivle_devices definition when cuda is not present
- dummy commit


## 1.3.0 (April 17, 2023)
- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.16 -> 1.3.0
- clean up pyproject.toml
- mv tests and rename imports in tests
- rename python package to parallelprojpy and adapt setup.cfg
- add first version of pyproject.toml
- dummy commit


## 1.2.16 (April 16, 2023)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.15 -> 1.2.16
- improve way to detect whether visible GPUs are present in the python API
- remove AS approximation of erff in openMP lib since it leads to too big inaccuracies
- add TOF LM tests
- add listmode wrappers
- add TOF sino fwd test
- use random image in TOF sino test
- dummy commit

## 1.2.15 (April 15, 2023)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.14 -> 1.2.15
- add TOF sino projector wrappers and first test
- BUGFIX: correct start and stop of loop over planes in cuda TOF sino projector when direction=2
- add adjointness test (indirect test for back projection)
- add first python unit test for non-tof fwd projection
- add first python wrappers for non-tof Joseph projectors
- start adding python wrappers
- remove recipe (moved to conda-forge feedstock)
- Update README.md
- dummy commit

## 1.2.14 (February 15, 2023)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.13 -> 1.2.14
- make target link libraries (m and OpenMP) private
- dummy commit

## 1.2.13 (January 13, 2023)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.12 -> 1.2.13
- fix variable expansion in Config.cmake.in
- update README
- add link to arxix preprint

## 1.2.12 (January 08, 2023)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.11 -> 1.2.12
- set CUDA_HOST_COMPILER only when using clang
- skip build of cuda lib if cuda is not present

## 1.2.11 (January 05, 2023)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.10 -> 1.2.11
- set default CMAKE_CUDA_HOST_COMPILER to CMAKE_CXX_COMPILER

## 1.2.10 (December 30, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.9 -> 1.2.10
- link parallelproj_c against libm (using PUBLIC link interface)
- use better way to test whether we have to link against libm
- Update README.md
- Merge pull request #18 from gschramm/add_generic_nontof_test
- updata non-tof cuda test
- add adjoint back projection test
- cosmetics
- add better output
- add more generic nontof test that tests rays in all 3 directions
- update README

## 1.2.9 (December 09, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.8 -> 1.2.9
- BUGFIX: correct calcultion of x_pr2 when principle direction is 0

## 1.2.8 (December 02, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.7 -> 1.2.8

## 1.2.7 (December 02, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.6 -> 1.2.7
- do not install test binaries
- require CXX compiler only for CUDA

## 1.2.6 (November 18, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.5 -> 1.2.6
- clean up CMake logic

## 1.2.5 (November 11, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.4 -> 1.2.5
- add conditions to nested if else when adding cuda subdir

## 1.2.4 (November 10, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.3 -> 1.2.4
- add fatal error if cuda lib is to be build but no cuda compiler is found
- add empty host sections to recipy on output level

## 1.2.3 (November 04, 2022)

- chore: updated package.json, updated CHANGELOG.md, bumped 1.2.2 -> 1.2.3
- work on local recipy and remove conda workflows
- add skip option for cmake

## 1.2.2 (November 03, 2022)

- chore: updated package.json, created CHANGELOG.md, bumped 1.2.1 -> 1.2.2
- read version from package.json (and use ver-bump) to bump version in the future
- remove python from new action
- add conda build
- add trigger
- add new conda action
