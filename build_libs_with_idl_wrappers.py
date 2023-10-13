# small wrapper script for all cmake calls
# to build all C and CUDA libs
# supposed to be OS independent

import argparse
import os
import platform
from tempfile import mkdtemp
from shutil import rmtree

parser = argparse.ArgumentParser(description = 'Build C/CUDA libs with cmake and install \
                                                them to the correct location for the python package')
parser.add_argument('--build_dir', help = 'temp build directory', 
                    default = None)
parser.add_argument('--source_dir', help = 'cmake source dir', 
                    default = os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--cmake_install_prefix', help = 'cmake INSTALL_LIB_DIR - default: %(default)s', 
                    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compiled_libs',f'{platform.system()}_{platform.architecture()[0]}'))
parser.add_argument('--keep_build_dir', help = 'do not remove tempory build dir', 
                    action = 'store_true')
parser.add_argument('--dry', help = 'dry run - only print cmake commands', 
                    action = 'store_true')
parser.add_argument('--cmake_bin', help = 'cmake binary to use', default = 'cmake') 
parser.add_argument('--generate_idl_wrappers', action = 'store_true')
parser.add_argument('--keep_idl_wrappers', help = 'do not remove tempory idl wrappers', 
                    action = 'store_true')
args = parser.parse_args()

#---------------------------------------------------------------------------------------------

if args.build_dir is None:
  build_dir  = mkdtemp(prefix = 'autobuild_', dir = '.')
else:
  build_dir = args.build_dir

source_dir            = args.source_dir
cmake_install_prefix  = args.cmake_install_prefix
remove_build_dir      = not args.keep_build_dir
dry                   = args.dry
cmake_bin             = args.cmake_bin
generate_idl_wrappers = args.generate_idl_wrappers
keep_idl_wrappers     = args.keep_idl_wrappers

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

def generate_idl_wrapper(src_file, wrapper_file, add_extern_C = False):
  """ Parse a C header file and generate idl wrappers for all included functions
  """
  from pyclibrary import CParser
  parser = CParser(src_file)

  if add_extern_C:
    extern_str = 'extern "C" '
  else:
    extern_str = ''

  with open(wrapper_file, 'w') as f:
    f.write('// AUTO_GENERATED - DO NOT MODIIFY\n\n')
    f.write(f'#include "{os.path.basename(src_file)}"\n\n')
    f.write('')
    
    for func_name,val in parser.defs['functions'].items():
      # write the function with return type
      rtype = ''.join(val[0])
      nargs = len(val[1])
    
      f.write(f'{extern_str}{rtype} {func_name}_idl_wrapper(int argc, void *argv[])\n')
      f.write('{\n')

      if rtype == 'void':
        rstatement = ''
      else:
        rstatement = 'return'
    
      # write the function arguments
      if len(val[1]) > 0:
        f.write(f'  {rstatement} {func_name}(\n')
        for i, args in enumerate(val[1]):
          type_qual = ''
          if 'const' in args[1].type_quals[0]: type_qual = 'const '
    

          if len(args[1]) == 3 and args[1][1] == '*' and args[1][2] == '*':
            prefix = f'({type_qual}{args[1][0]}**)'
          elif len(args[1]) == 2 and args[1][1] == '*':
            prefix = f'({type_qual}{args[1][0]}*)'
          else:
            prefix = f'*({type_qual}{args[1][0]}*)'
        
          if i < (nargs - 1):
            f.write(f'    {prefix} argv[{i}],\n')
          else:
            f.write(f'    {prefix} argv[{i}]);\n')

        f.write('}\n')
        f.write('\n')
      else:
          f.write(f'  {rstatement} {func_name}();\n}}\n')

  print(f'generated {wrapper_file}')

#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

# generate the IDL wrappers
if generate_idl_wrappers:
  c_idl_wrapper_dir = os.path.join('c','wrapper')
  os.makedirs(c_idl_wrapper_dir, exist_ok = True)
  header_file  = os.path.join('c','include','parallelproj_c.h')
  wrapper_file = os.path.join('c','wrapper',
                        f'{os.path.splitext(os.path.basename(header_file))[0]}_idl_wrapper.c')
  generate_idl_wrapper(header_file, wrapper_file)

  cuda_idl_wrapper_dir = os.path.join('cuda','wrapper')
  os.makedirs(cuda_idl_wrapper_dir, exist_ok = True)
  header_file_cuda  = os.path.join('cuda','include','parallelproj_cuda.h')
  wrapper_file_cuda = os.path.join('cuda','wrapper',
                        f'{os.path.splitext(os.path.basename(header_file_cuda))[0]}_idl_wrapper.cu')
  generate_idl_wrapper(header_file_cuda, wrapper_file_cuda, add_extern_C = True)


#---------------------------------------------------------------------------------------------

# on windows DLLs get install in CMAKE_INSTALL_BINDIR
cmake_options = f'-B {build_dir} -DCMAKE_INSTALL_PREFIX={cmake_install_prefix}'

if generate_idl_wrappers:
  cmake_options = f'{cmake_options} -DPARALLELPROJ_BUILD_WITH_IDL_WRAPPERS=TRUE'

cmd1 = f'{cmake_bin} {cmake_options}  {source_dir}'
cmd2 = f'{cmake_bin} --build {build_dir} --target install --config release'

if dry:
  print(cmd1,'\n')
  print(cmd2)
else:
  os.system(cmd1)
  os.system(cmd2)

if remove_build_dir:
  rmtree(build_dir)
else:
  print(f'Kept build directory {build_dir}')

if generate_idl_wrappers:
  if not keep_idl_wrappers:
    rmtree(c_idl_wrapper_dir)
    rmtree(cuda_idl_wrapper_dir)
  else:
    print(f'Kept idl c wrapper directory {c_idl_wrapper_dir}')
    print(f'Kept idl cuda wrapper directory {cuda_idl_wrapper_dir}')
