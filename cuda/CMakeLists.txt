enable_language(CUDA)

if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  file(GLOB CUDA_SOURCES src/*.cu wrapper/*.cu)
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

# we explicitly specify DESTINATION here, because otherwise DLLs on Windows
# are installed in ${CMAKE_INSTALL_BINDIR}
install(TARGETS parallelproj_cuda EXPORT parallelprojTargets DESTINATION ${PARALLELPROJ_INSTALL_LIBDIR} PUBLIC_HEADER DESTINATION ${PARALLELPROJ_INSTALL_INCLUDEDIR})