add_executable(parallelproj_nontof_fwd_back_cuda_test nontof_fwd_back_cuda.c)
target_include_directories(parallelproj_nontof_fwd_back_cuda_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
target_link_libraries(parallelproj_nontof_fwd_back_cuda_test PUBLIC parallelproj_cuda)

# add CUDA test - only if we have a GPU
execute_process(COMMAND nvidia-smi --query-gpu=name --format=csv,noheader
                RESULT_VARIABLE GPU_RESULT
                OUTPUT_VARIABLE GPU_OUTPUT)

message(STATUS "GPUS RES: ${GPU_RESULT}")
# If there are not GPUs available (or nvidia-smi is not available), GPU_OUTPUT is an empty string
message(STATUS "Detected GPUS: ${GPU_OUTPUT}")

if ("${GPU_OUTPUT}" STREQUAL "")
  message(WARNING "NO GPUs found, skipping cuda tests")
else()
  add_test(NAME parallelproj_nontof_fwd_back_cuda_test COMMAND parallelproj_nontof_fwd_back_cuda_test)
endif()
