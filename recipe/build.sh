cmake -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib ${CMAKE_ARGS} ${SRC_DIR}
cmake --build . --target install
