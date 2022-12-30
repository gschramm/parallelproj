add_executable(parallelproj_nontof_fwd_back_test nontof_fwd_back.c)
target_include_directories(parallelproj_nontof_fwd_back_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
target_link_libraries(parallelproj_nontof_fwd_back_test PUBLIC parallelproj_c)

add_test(NAME parallelproj_nontof_fwd_back_test COMMAND parallelproj_nontof_fwd_back_test)