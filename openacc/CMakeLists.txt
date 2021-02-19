file(GLOB C_SOURCES src/*.c)
add_library(parallelproj_openacc SHARED ${C_SOURCES})
add_library(parallelproj::parallelproj_openacc ALIAS parallelproj_openacc)
target_include_directories(parallelproj_openacc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>
  )
set_target_properties(parallelproj_openacc PROPERTIES PUBLIC_HEADER "include/parallelproj_openacc.h")

target_compile_options(parallelproj_openacc PRIVATE ${OpenACC_C_OPTIONS})

install(TARGETS parallelproj_openacc EXPORT parallelprojTargets)
