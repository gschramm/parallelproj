# structure and content of CMakeLists.txt files adapted from 
# https://stackoverflow.com/questions/55635294/how-to-create-packages-with-cmake

cmake_minimum_required(VERSION 3.16.0)

message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")

include(CMakePackageConfigHelpers)
include(CheckLanguage)

#-------------------------------------------------------------
# Read the line containing 'version =' from pyproject.toml
file(STRINGS pyproject.toml versionLine REGEX "^version = ")

# Extract the version number from the matched line
if("${versionLine}" MATCHES "^version = \"([0-9]+\\.[0-9]+\\.[0-9]+)\"")
  set(PARALLELPROJ_VERSION "${CMAKE_MATCH_1}")
  message(STATUS "BUILDING VERSION: ${PARALLELPROJ_VERSION}")
else()
  message(FATAL_ERROR "Version not found in pyproject.toml")
endif()

#-------------------------------------------------------------

project(parallelproj VERSION ${PARALLELPROJ_VERSION} LANGUAGES C)

include(CTest)
enable_testing()

# get standard paths for installation
include(GNUInstallDirs)

set (CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)

#-------------------------------------------------------------

# define INSTALL dirs for libraries, public headers and documentation
# by default we use the GNUInstallDirs

# variable whether to build with auto-generated IDL wrappers
set(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS FALSE CACHE BOOL "include auto-generated IDL wrapper sources in build")

set(SKIP_OPENMP_LIB FALSE CACHE BOOL "do not build the openmp lib")
set(SKIP_CUDA_LIB FALSE CACHE BOOL "do not build the cuda lib")
set(SKIP_DOCS FALSE CACHE BOOL "do not build the documentation")

# skip build of the CUDA lib, if CUDA is not available
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA not available skipping build of CUDA LIB")
  set(SKIP_CUDA_LIB TRUE)
endif()

#-------------------------------------------------------------

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

#-------------------------------------------------------------


# check if Doxygen is installed
find_package(Doxygen)


#-------------------------------------------------------------
# build the C/OpenMP lib
if (NOT SKIP_OPENMP_LIB)
  add_subdirectory(c)
else()
  message(STATUS "Skipping build of openMP lib")
endif()


if(NOT SKIP_CUDA_LIB)
  add_subdirectory(cuda)
else(NOT SKIP_CUDA_LIB)
  message(STATUS "Skipping build of cuda lib")
endif(NOT SKIP_CUDA_LIB)

#-------------------------------------------------------------
# install the targets
install(
  EXPORT parallelprojTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
  NAMESPACE parallelproj::
  FILE parallelprojTargets.cmake # Not sure if this is still needed
  )


#-------------------------------------------------------------
# build the documentation with Doxygen

if (NOT SKIP_DOCS)
  if (DOXYGEN_FOUND)
    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
  
    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
  
    # note the option ALL which allows to build the docs together with the application
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )
  endif (DOXYGEN_FOUND)
else()
  message(STATUS "skipping build of documentation")
endif()

#-------------------------------------------------------------
# generate and install cmake package and version files
  
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj
  PATH_VARS
    CMAKE_INSTALL_LIBDIR
  )

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfigVersion.cmake
  VERSION ${parallelproj_VERSION}
  COMPATIBILITY SameMajorVersion
  )

# Install Config and ConfigVersion files
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/parallelprojConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/parallelproj"
  )

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
