cmake_minimum_required(VERSION 3.9.0)

set (CMAKE_BUILD_TYPE Release)

project(pynucmed_proj LANGUAGES C CUDA)

# (1) build the standard C lib with OpenMP support
# find the required OpeMP
find_package(OpenMP REQUIRED)

file(GLOB CP_SOURCES src/*.c)
add_library(parallelproj SHARED ${CP_SOURCES})
target_include_directories(parallelproj PRIVATE include)
if(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(parallelproj PRIVATE m)
endif(CMAKE_COMPILER_IS_GNUCC)
# we have to add the openmp library to be linked
target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_C)
install(TARGETS parallelproj DESTINATION ${CMAKE_CURRENT_LIST_DIR}/lib)



# (2) build the cuda lib
# set cuda compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options '-fPIC'")

file(GLOB CP_SOURCES src/*.cu)
add_library(parallelproj_cuda SHARED ${CP_SOURCES})
#target_include_directories(parallelproj_cuda PRIVATE include)
install(TARGETS parallelproj_cuda DESTINATION ${CMAKE_CURRENT_LIST_DIR}/lib)
