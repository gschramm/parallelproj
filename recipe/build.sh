cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE="Release" ${SRC_DIR}

cmake --build . --target install
