# structure and content of CMakeLists.txt files adapted from 
# https://stackoverflow.com/questions/55635294/how-to-create-packages-with-cmake

cmake_minimum_required(VERSION 3.9.0)
project(parallelproj LANGUAGES C CXX VERSION 1.0.0)

include(CMakePackageConfigHelpers)
include(CheckLanguage)

# get standard paths for installation
include(GNUInstallDirs)

set (CMAKE_BUILD_TYPE Release CACHE STRING "build type" FORCE)

#-------------------------------------------------------------

# define INSTALL dirs for libraries, public headers and documentation
# by default we use the GNUInstallDirs

# variable whether to build with auto-generated IDL wrappers
set(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS FALSE CACHE BOOL "include auto-generated IDL wrapper sources in build")
#-------------------------------------------------------------

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)

#-------------------------------------------------------------
# checks for OpenMP, CUDA and doxygen

# find the required OpeMP
find_package(OpenMP REQUIRED)

# check if CUDA is available
check_language(CUDA)

# check if Doxygen is installed
find_package(Doxygen)


#-------------------------------------------------------------
# build the C/OpenMP lib
add_subdirectory(c)


# only build CUDA lib if CUDA is available
if(CMAKE_CUDA_COMPILER)
  add_subdirectory(cuda)
endif()

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


#-------------------------------------------------------------
# generate and install cmake package and version files
if(CMAKE_CUDA_COMPILER)
  set(CUDA_SUPPORTED ON)
endif()
  
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

#packaging
if (WIN32)
  # requires build of Release
  # cmake --build . --config Release
  set(CPACK_GENERATOR NSIS)
else()
  set(CPACK_GENERATOR "RPM;DEB")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Georg Schramm")
endif()
include(CPack)
