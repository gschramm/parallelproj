file(GLOB C_SOURCES src/*.c)
add_library(parallelproj_c SHARED ${C_SOURCES})
add_library(parallelproj::parallelproj_c ALIAS parallelproj_c)
target_include_directories(parallelproj_c PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
set_target_properties(parallelproj_c PROPERTIES PUBLIC_HEADER "include/parallelproj_c.h")
# we have to add the openmp library to be linked
target_link_libraries(parallelproj_c PUBLIC OpenMP::OpenMP_C)

install(TARGETS parallelproj_c EXPORT parallelprojTargets
ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
