add_executable(nontof_fwd_back nontof_fwd_back.c)
target_include_directories(nontof_fwd_back PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
target_link_libraries(nontof_fwd_back PUBLIC parallelproj_c)

# check if we need to link against libm (we need to do that e.g. for gcc, but not for msvc)
set(POW_LIBS "")
include(CheckLibraryExists)
check_library_exists(m pow "" LIBM)
if(LIBM)
    message(STATUS "linking test against libm")
    list(APPEND POW_LIBS "m")
endif()

if(CMAKE_COMPILER_IS_GNUCC)
  target_link_libraries(nontof_fwd_back PUBLIC ${POW_LIBS})
endif()

add_test(NAME nontof_fwd_back COMMAND nontof_fwd_back)
