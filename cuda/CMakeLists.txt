enable_language(CXX)

# set the CUDA host compiler by default to the CMAKE CXX COMPILER if clang is used
# e.g. important for setting the openMP flags for the host compiler correctly
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
    message(STATUS "CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER}")
  endif()
endif()


enable_language(CUDA)

# find the required OpeMP
find_package(OpenMP REQUIRED)

# auto detect the cuda architecture
# https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures
# for cmake >= 3.23 it is better to use CMAKE_CUDA_ARCHITECTURES=all or native, but we want to support older cmake
# versions as well
# if we build on a system without GPU, it is best to just set CMAKE_CUDA_ARCHITECUTURES to e.g. 75
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(${CMAKE_VERSION} VERSION_LESS "3.23.0")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES NOT DEFINED")
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "all")
  endif()
endif()

message(STATUS "Using CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# add openmp flags to cuda compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}") 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}") 

if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  file(GLOB CUDA_SOURCES src/*.cu wrapper/parallelproj_cuda_idl_wrapper.cu)
else()
  file(GLOB CUDA_SOURCES src/*.cu)
endif()

add_library(parallelproj_cuda SHARED ${CUDA_SOURCES})
add_library(parallelproj::parallelproj_cuda ALIAS parallelproj_cuda)
set_target_properties(parallelproj_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(parallelproj_cuda PROPERTIES PUBLIC_HEADER "include/parallelproj_cuda.h"
                                                   SOVERSION ${parallelproj_VERSION})

target_include_directories(parallelproj_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )

# we have to add the openmp library to be linked
target_link_libraries(parallelproj_cuda PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS parallelproj_cuda EXPORT parallelprojTargets 
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install the cuda kernel source as well which could be used by e.g. cupy
install(FILES "src/projector_kernels.cu" RENAME "projector_kernels.cu.${PARALLELPROJ_VERSION}" DESTINATION ${CMAKE_INSTALL_LIBDIR})

# install auto-generated IDL wrapper
if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  install(FILES "wrapper/parallelproj_cuda_idl_wrapper.cu" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# add test executables
add_subdirectory(tests)

