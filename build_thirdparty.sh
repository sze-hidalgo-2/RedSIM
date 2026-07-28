#!/bin/bash
echo "thirdparty builds started"

# NOTE(cmat): Exit on error.
set -eu

# NOTE(cmat): Set working directory to the build.sh folder.
cd "$(dirname "$0")"

build_folder="build"
mkdir -p $build_folder
pushd $build_folder > /dev/null 2>&1

# NOTE(cmat): Build Zoltan from source, as a static library.
../src/thirdparty/zoltan/build_zoltan.sh
mv ../src/thirdparty/zoltan/libzoltan.a .

popd > /dev/null 2>&1

echo "thirdparty build completed"

