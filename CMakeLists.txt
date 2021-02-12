cmake_minimum_required(VERSION 3.9.0)
include(CMakePackageConfigHelpers)
include(CheckLanguage)

project(parallelproj LANGUAGES C)
set (CMAKE_BUILD_TYPE Release)

#-------------------------------------------------------------
# define the version number
set(VERSION_MAJOR   0)
set(VERSION_MINOR   1)
set(VERSION_PATCH   0-prealpha)

set(parallelproj_VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})


#-------------------------------------------------------------
# define variables

# directory where libs get installed
set(INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib/parallelproj CACHE PATH "install directory for compiled libs")
# directory where cmake targets get installed
set(CONFIG_FILE_DIR ${CMAKE_INSTALL_PREFIX}/lib/cmake CACHE PATH "install directory for cmake targets")

# define include dirs
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/include/c)
set(INCLUDE_DIR_CUDA ${CMAKE_CURRENT_SOURCE_DIR}/src/include/cuda)


#-------------------------------------------------------------
# checks for OpenMP, CUDA and doxygen

# find the required OpeMP
find_package(OpenMP REQUIRED)
add_definitions(${OpenMP_C_FLAGS})

# check if CUDA is available
check_language(CUDA)

# check if Doxygen is installed
find_package(Doxygen)



#-------------------------------------------------------------
# (1) build the standard C lib with OpenMP support

file(GLOB C_SOURCES src/c/*.c)
add_library(parallelproj SHARED ${C_SOURCES})
target_include_directories(parallelproj PRIVATE ${INCLUDE_DIR})
# we have to add the openmp library to be linked
target_link_libraries(parallelproj PRIVATE OpenMP::OpenMP_C)
install(TARGETS parallelproj DESTINATION ${INSTALL_LIB_DIR} EXPORT parallelproj-targets)



#-------------------------------------------------------------
# (2) build the cuda lib

# only build CUDA lib if CUDA is available
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  file(GLOB CUDA_SOURCES src/cuda/*.cu)
  add_library(parallelproj_cuda SHARED ${CUDA_SOURCES})
  set_target_properties(parallelproj_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_include_directories(parallelproj_cuda PRIVATE ${INCLUDE_DIR_CUDA})
  install(TARGETS parallelproj_cuda DESTINATION ${INSTALL_LIB_DIR} EXPORT parallelproj-targets)
endif()



#-------------------------------------------------------------
# package config and version file

configure_package_config_file(
  src/cmake/parallelprojConfig.cmake.in
  "${CMAKE_BINARY_DIR}/parallelprojConfig.cmake"
   INSTALL_DESTINATION "${CONFIG_FILE_DIR}"
   PATH_VARS INCLUDE_DIR INCLUDE_DIR_CUDA)

write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/parallelprojVersion.cmake"
  VERSION ${parallelproj_VERSION}
  COMPATIBILITY SameMajorVersion )

#-------------------------------------------------------------
# install the config files

# export the targets
install(EXPORT parallelproj-targets DESTINATION ${CONFIG_FILE_DIR})

# install STIRConfig*.cmake
install(
  FILES
    "${CMAKE_BINARY_DIR}/parallelprojConfig.cmake"
    "${CMAKE_BINARY_DIR}/parallelprojVersion.cmake"
  DESTINATION "${CONFIG_FILE_DIR}"
)



#-------------------------------------------------------------
# (3) build the documentation with Doxygen

if (DOXYGEN_FOUND)
  # set input and output files
  set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # request to configure the file
  configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

  # note the option ALL which allows to build the docs together with the application
  add_custom_target( doc_doxygen ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM )
else (DOXYGEN_FOUND)
  message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
