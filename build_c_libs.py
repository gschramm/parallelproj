# small wrapper script for all cmake calls
# to build all C and CUDA libs
# supposed to be OS independent

import os
from tempfile import TemporaryDirectory

build_dir  = TemporaryDirectory(prefix = 'build', dir = '.')
source_dir = '.'

if os.name == 'nt':
  os.system(f'cmake -B {build_dir.name} -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE {source_dir}')
  os.system(f'cmake --build {build_dir.name} --target INSTALL --config RELEASE')
else:
  os.system(f'cmake -B {build_dir.name} {source_dir}')
  os.system(f'cmake --build {build_dir.name}')
  os.system(f'cmake --install {build_dir.name}')
