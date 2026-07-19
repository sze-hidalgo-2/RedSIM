#!/bin/bash
echo "build started"

# NOTE(cmat): Exit on error.
set -eu

# NOTE(cmat): Unpack arguments.
for arg in "$@"; do
  declare $arg='1';
done

# NOTE(cmat): Generate tags file if ctags is available.
if command -v ctags &> /dev/null; then
  ctags -R .
fi

# NOTE(cmat): Set working directory to the build.sh folder.
cd "$(dirname "$0")"

# NOTE(cmat): Absolute path for source files (compiler errors are cleaner).
source_folder=$(realpath "./src")

# NOTE(cmat): Compilation parameters.
build_folder="build"

include_folders="-I${source_folder}"

define_flags=""

compiler_exec=""
source_files=""
compiler_flags=""
linker_flags=""

# NOTE(cmat): Entry point.
source_files+=" ${source_folder}/redsim/rs_entry.c"

# ------------------------------------------------------------
# -- Clean build
mkdir -p $build_folder

compiler_exec+="mpicc"
define_flags+=" -D_GNU_SOURCE"

if [[ -n ${release-} ]]; then
  echo "release: yes"
  compiler_flags+=" -O3 -flto"
  linker_flags+=" -flto"
  define_flags+=" -DBUILD_ASSERT=0"
else
  echo "release: no"
  compiler_flags+=" -O0"
  compiler_flags+=" -fsanitize=address"
  compiler_flags+=" -fsanitize-address-use-after-scope"
  compiler_flags+=" -fno-omit-frame-pointer"
  define_flags+=" -DBUILD_ASSERT=1"
fi

if [[ -n ${no_debug-} ]]; then
  echo "debug: no"
else
  echo "debug: yes"
  compiler_flags+=" -g"
fi

if [[ -n ${no_profile-} ]]; then
  echo "profile: no"
  define_flags+=" -DBUILD_PROFILE=0"
else
  echo "profile: yes"
  define_flags+=" -DBUILD_PROFILE=1"
fi

if [[ -n ${fast_math-} ]]; then
echo "fast_math: yes"
compiler_flags+=" -ffast-math"
compiler_flags+=" -Wno-nan-infinity-disabled"
else
  echo "fast_math: no"
fi

compiler_flags+=" -march=native"
compiler_flags+=" -std=c99"
# compiler_flags+=" -Wall"

linker_flags+=" -lnuma"
linker_flags+=" -lpthread"
linker_flags+=" -o redsim_cpu"

# ------------------------------------------------------------
# NOTE(cmat): Invoke compiler
pushd build > /dev/null 2>&1

echo "compiler flags: ${compiler_flags}"
echo "linker   flags: ${linker_flags}"
$compiler_exec $source_files $define_flags $include_folders $compiler_flags $linker_flags

popd > /dev/null 2>&1

echo "build completed"
