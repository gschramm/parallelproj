enable_language(CUDA)

# add openmp flags to cuda compiler
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_C_FLAGS}") 
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}") 

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
target_link_libraries(parallelproj_cuda PUBLIC OpenMP::OpenMP_C)

install(TARGETS parallelproj_cuda EXPORT parallelprojTargets 
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install auto-generated IDL wrapper
if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  install(FILES "wrapper/parallelproj_cuda_idl_wrapper.cu" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
