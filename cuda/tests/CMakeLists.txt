add_executable(nontof_fwd_back_cuda nontof_fwd_back_cuda.c)
target_include_directories(nontof_fwd_back_cuda PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
target_link_libraries(nontof_fwd_back_cuda PUBLIC parallelproj_cuda)

# add CUDA test - only if we have a GPU
execute_process(COMMAND nvidia-smi --query-gpu=name --format=csv,noheader
                RESULT_VARIABLE NUM_GPU_RESULT
                OUTPUT_VARIABLE NUM_GPU_OUTPUT)

message(INFO "NUM GPUS RES: ${NUM_GPU_RESULT}")
message(INFO "NUM GPUS: ${NUM_GPU_OUTPUT}")

add_test(NAME nontof_fwd_back_cuda COMMAND nontof_fwd_back_cuda)
