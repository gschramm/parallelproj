enable_language(CUDA)

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

install(TARGETS parallelproj_cuda EXPORT parallelprojTargets 
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install auto-generated IDL wrapper
if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  install(FILES "wrapper/parallelproj_cuda_idl_wrapper.cu" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
