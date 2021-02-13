# small wrapper script for all cmake calls
# to build all C and CUDA libs
# supposed to be OS independent

import argparse
import os
from tempfile import mkdtemp
from shutil import rmtree

parser = argparse.ArgumentParser(description = 'Build C/CUDA libs with cmake and install \
                                                them to the correct location for the python package')
parser.add_argument('--build_dir', help = 'temp build directory', 
                    default = None)
parser.add_argument('--source_dir', help = 'cmake source dir', 
                    default = os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--cmake_install_prefix', help = 'cmake INSTALL_LIB_DIR - default: %(default)s', 
                    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyparallelproj'))
parser.add_argument('--keep_build_dir', help = 'do not remove tempory build dir', 
                    action = 'store_true')
parser.add_argument('--dry', help = 'dry run - only print cmake commands', 
                    action = 'store_true')
args = parser.parse_args()

#---------------------------------------------------------------------------------------------

if args.build_dir is None:
  build_dir  = mkdtemp(prefix = 'build_', dir = '.')
else:
  build_dir = args.build_dir

source_dir = args.source_dir

cmake_install_prefix = args.cmake_install_prefix

remove_build_dir = not args.keep_build_dir

dry = args.dry

#---------------------------------------------------------------------------------------------

if os.name == 'nt':
  cmd1 = f'cmake -B {build_dir} -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DCMAKE_INSTALL_PREFIX={cmake_install_prefix} {source_dir}'
  cmd2 = f'cmake --build {build_dir} --target INSTALL --config RELEASE'

  if dry:
    print(cmd1,'\n')
    print(cmd2)
  else:
    os.system(cmd1)
    os.system(cmd2)
else:
  cmd1 = f'cmake -B {build_dir} -DCMAKE_INSTALL_PREFIX={cmake_install_prefix} {source_dir}'
  cmd2 = f'cmake --build {build_dir}'
  cmd3 = f'cmake --install {build_dir}'

  if dry:
    print(cmd1,'\n')
    print(cmd2,'\n')
    print(cmd3)
  else:
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)

if remove_build_dir:
  rmtree(build_dir)
else:
  print(f'Kept build directory {build_dir}')

