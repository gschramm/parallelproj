if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  file(GLOB C_SOURCES src/*.c wrapper/*.c)
else()
  file(GLOB C_SOURCES src/*.c)
endif()

add_library(parallelproj_c SHARED ${C_SOURCES})
add_library(parallelproj::parallelproj_c ALIAS parallelproj_c)
target_include_directories(parallelproj_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
set_target_properties(parallelproj_c PROPERTIES PUBLIC_HEADER "include/parallelproj_c.h"
                                                SOVERSION ${parallelproj_VERSION})

# we have to add the openmp library to be linked
target_link_libraries(parallelproj_c PUBLIC OpenMP::OpenMP_C)

# we explicitly specify DESTINATION here, because otherwise DLLs on Windows
# are installed in ${CMAKE_INSTALL_BINDIR}
install(TARGETS parallelproj_c EXPORT parallelprojTargets 
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
