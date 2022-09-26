add_executable(nontof_fwd_back_cuda nontof_fwd_back_cuda.c)
target_include_directories(nontof_fwd_back_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
target_link_libraries(nontof_fwd_back_cuda PUBLIC parallelproj_cuda)

add_test(NAME nontof_fwd_back_cuda COMMAND nontof_fwd_back_cuda)
set_property(TEST nontof_fwd_back_cuda PROPERTY RESOURCE_GROUPS "gpus:1")
