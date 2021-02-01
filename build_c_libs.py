# small wrapper script for all cmake calls
# to build all C and CUDA libs
# supposed to be OS independent

import os
from tempfile import TemporaryDirectory

build_dir  = TemporaryDirectory(prefix = 'build', dir = '.')
source_dir = '.'

if os.name == 'nt':
  cmake_args = '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE'
else:
  cmake_args = ''

os.system(f'cmake -B {build_dir.name} {cmake_args} {source_dir}')
os.system(f'cmake --build {build_dir.name}')
os.system(f'cmake --install {build_dir.name}')
