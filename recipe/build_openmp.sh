cmake -GNinja -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DSKIP_DOCS=TRUE -DSKIP_CUDA_LIB=TRUE ${CMAKE_ARGS} ${SRC_DIR}
cmake --build . --target install --verbose
ctest -VV