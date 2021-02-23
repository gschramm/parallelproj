enable_language(CUDA)
file(GLOB CUDA_SOURCES src/*.cu)
add_library(parallelproj_cuda SHARED ${CUDA_SOURCES})
add_library(parallelproj::parallelproj_cuda ALIAS parallelproj_cuda)
set_target_properties(parallelproj_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(parallelproj_cuda PROPERTIES PUBLIC_HEADER "include/parallelproj_cuda.h")
target_include_directories(parallelproj_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )

# we explicitly specify DESTINATION here, because otherwise DLLs on Windows
# are installed in ${CMAKE_INSTALL_BINDIR}
install(TARGETS parallelproj_cuda EXPORT parallelprojTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parallelproj_cuda PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
