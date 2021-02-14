file(GLOB C_SOURCES src/*.c)
add_library(parallelproj_c SHARED ${C_SOURCES})
add_library(parallelproj::parallelproj_c ALIAS parallelproj_c)
target_include_directories(parallelproj_c PRIVATE include)
set_target_properties(parallelproj_c PROPERTIES PUBLIC_HEADER "include/parallelproj_c.h")
# we have to add the openmp library to be linked
target_link_libraries(parallelproj_c PRIVATE OpenMP::OpenMP_C)

install(TARGETS parallelproj_c EXPORT parallelprojTargets
ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
