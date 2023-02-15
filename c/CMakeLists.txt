enable_language(C)

# find the required OpeMP
find_package(OpenMP REQUIRED)

if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  file(GLOB C_SOURCES src/*.c wrapper/parallelproj_c_idl_wrapper.c)
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

# link to libm if it exists and against OpenMP_C
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    message(STATUS "linking against libm")
    target_link_libraries(parallelproj_c PRIVATE ${MATH_LIBRARY} OpenMP::OpenMP_C)
else()
    target_link_libraries(parallelproj_c PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS parallelproj_c EXPORT parallelprojTargets 
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install auto-generated IDL wrapper
if(PARALLELPROJ_BUILD_WITH_IDL_WRAPPERS)
  install(FILES "wrapper/parallelproj_c_idl_wrapper.c" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# add the test executables
add_subdirectory(tests)

