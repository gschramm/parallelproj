cmake_minimum_required(VERSION 3.9.0)

set (CMAKE_BUILD_TYPE Release)

project(parallelproj LANGUAGES C)

# (1) build the standard C lib with OpenMP support
# find the required OpeMP
find_package(OpenMP REQUIRED)

file(GLOB CP_SOURCES src/*.c)
add_library(parallelproj SHARED ${CP_SOURCES})
target_include_directories(parallelproj PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
if(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(parallelproj PRIVATE m)
endif(CMAKE_COMPILER_IS_GNUCC)
# we have to add the openmp library to be linked
target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_C)
install(TARGETS parallelproj DESTINATION ${CMAKE_CURRENT_LIST_DIR}/pyparallelproj/lib EXPORT parallelproj-targets)

# (2) build the cuda lib
include(CheckLanguage)
check_language(CUDA)

# only build CUDA lib if CUDA is available
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  # set cuda compiler flags
  #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options '-fPIC'")
  file(GLOB CP_SOURCES src/*.cu)
  add_library(parallelproj_cuda SHARED ${CP_SOURCES})
  set_target_properties(parallelproj_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(parallelproj_cuda PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include)
  install(TARGETS parallelproj_cuda DESTINATION ${CMAKE_CURRENT_LIST_DIR}/pyparallelproj/lib EXPORT parallelproj-targets)
else()
  message("XXXX NO CUDA FOUND -> skipping CUDA lib")
endif()

# export the targets
install(EXPORT parallelproj-targets DESTINATION ${CMAKE_CURRENT_LIST_DIR}/pyparallelproj/lib)


# (3) build the documentation with Doxygen

# check if Doxygen is installed
find_package(Doxygen)
if (DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  message("Doxygen build started")

  # note the option ALL which allows to build the docs together with the application
  add_custom_target( doc_doxygen ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
