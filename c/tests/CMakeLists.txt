add_executable(parallelproj_nontof_fwd_back_test nontof_fwd_back.c)
target_include_directories(parallelproj_nontof_fwd_back_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
target_link_libraries(parallelproj_nontof_fwd_back_test PUBLIC parallelproj_c)

# check if we need to link against libm (we need to do that e.g. for gcc, but not for msvc)
set(POW_LIBS "")
include(CheckLibraryExists)
check_library_exists(m pow "" LIBM)
if(LIBM)
    message(STATUS "linking test against libm")
    list(APPEND POW_LIBS "m")
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(parallelproj_nontof_fwd_back_test PUBLIC ${POW_LIBS})
endif()

install(TARGETS parallelproj_nontof_fwd_back_test EXPORT parallelprojTargets 
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


add_test(NAME parallelproj_nontof_fwd_back_test COMMAND parallelproj_nontof_fwd_back_test)
