add_executable(nontof_fwd_back nontof_fwd_back.c)
target_include_directories(nontof_fwd_back PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
target_link_libraries(nontof_fwd_back PUBLIC parallelproj_c)

add_test(NAME nontof_fwd_back COMMAND nontof_fwd_back)