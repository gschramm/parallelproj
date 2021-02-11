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
parser.add_argument('--install_dir', help = 'cmake INSTALL_LIB_DIR - default: %(default)s', 
                    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'pyparallelproj','lib'))
parser.add_argument('--target_dir', help = 'cmake INSTALL_TARGET_DIR - default: %(default)s', 
                    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'pyparallelproj','lib'))
parser.add_argument('--keep_build_dir', help = 'do not remove tempory build dir', 
                    action = 'store_true')
args = parser.parse_args()

#---------------------------------------------------------------------------------------------

if args.build_dir is None:
  build_dir  = mkdtemp(prefix = 'build_', dir = '.')
else:
  build_dir = args.build_dir

source_dir = args.source_dir

install_dir = args.install_dir

target_dir = args.target_dir

remove_build_dir = not args.keep_build_dir

#---------------------------------------------------------------------------------------------

if os.name == 'nt':
  os.system(f'cmake -B {build_dir} -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=TRUE -DINSTALL_LIB_DIR={install_dir} -DINSTALL_TARGET_DIR={target_dir} {source_dir}')
  os.system(f'cmake --build {build_dir} --target INSTALL --config RELEASE')
else:
  os.system(f'cmake -B {build_dir} -DINSTALL_LIB_DIR={install_dir} -DINSTALL_TARGET_DIR={target_dir}  {source_dir}')
  os.system(f'cmake --build {build_dir}')
  os.system(f'cmake --install {build_dir}')

if remove_build_dir:
  rmtree(build_dir)
else:
  print(f'Kept build directory {build_dir}')

