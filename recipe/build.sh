CMAKE_PLATFORM_FLAGS+=(-DCMAKE_TOOLCHAIN_FILE="${RECIPE_DIR}/cross-linux.cmake")

cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} ${CMAKE_PLATFORM_FLAGS[@]} -DCMAKE_BUILD_TYPE="Release" ${SRC_DIR}

#cmake --build . --target install
make VERBOSE=2
