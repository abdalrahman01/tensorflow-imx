#!/usr/bin/env bash
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
# Copyright 2021 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Usage: CI_BUILD_PYTHON=python ../tensorflow/lite/tools/pip_package/build_pip_package_with_cmake2.sh aarch64 
set -ex
echo ${PYTHON}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CI_BUILD_PYTHON:-python3}"
VERSION_SUFFIX=${VERSION_SUFFIX:-}
export TENSORFLOW_DIR="${SCRIPT_DIR}/../../../.."
TENSORFLOW_LITE_DIR="${TENSORFLOW_DIR}/tensorflow/lite"
TENSORFLOW_VERSION=$(grep "_VERSION = " "${TENSORFLOW_DIR}/tensorflow/tools/pip_package/setup.py" | cut -d= -f2 | sed "s/[ '-]//g")
export PACKAGE_VERSION="${TENSORFLOW_VERSION}${VERSION_SUFFIX}"
BUILD_DIR=`pwd`
TENSORFLOW_TARGET=$1

mkdir -p "${BUILD_DIR}/tflite_pip"
mkdir -p "${BUILD_DIR}/tflite_pip/tflite_runtime"
cp -r "${TENSORFLOW_LITE_DIR}/tools/pip_package/debian" \
      "${TENSORFLOW_LITE_DIR}/tools/pip_package/setup_with_binary.py" \
      "${TENSORFLOW_LITE_DIR}/tools/pip_package/MANIFEST.in" \
      "${TENSORFLOW_LITE_DIR}/python/interpreter_wrapper" \
      "${BUILD_DIR}/tflite_pip"
cp "${TENSORFLOW_LITE_DIR}/python/interpreter.py" \
   "${TENSORFLOW_LITE_DIR}/python/metrics_interface.py" \
   "${TENSORFLOW_LITE_DIR}/python/metrics_portable.py" \
   "${BUILD_DIR}/tflite_pip/tflite_runtime"
echo "__version__ = '${PACKAGE_VERSION}'" >> "${BUILD_DIR}/tflite_pip/tflite_runtime/__init__.py"
echo "__git_version__ = '$(git -C "${TENSORFLOW_DIR}" describe)'" >> "${BUILD_DIR}/tflite_pip/tflite_runtime/__init__.py"
# Build python interpreter_wrapper.

cmake --build . --verbose -j ${BUILD_NUM_JOBS} -t _pywrap_tensorflow_interpreter_wrapper
cd "${BUILD_DIR}"

case "${TENSORFLOW_TARGET}" in
  windows)
    LIBRARY_EXTENSION=".pyd"
    ;;
  *)
    LIBRARY_EXTENSION=".so"
    ;;
esac

cp "${BUILD_DIR}/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}" \
   "${BUILD_DIR}/tflite_pip/tflite_runtime"
# Bazel generates the wrapper library with r-x permissions for user.
# At least on Windows, we need write permissions to delete the file.
# Without this, setuptools fails to clean the build directory.
chmod u+w "${BUILD_DIR}/tflite_pip/tflite_runtime/_pywrap_tensorflow_interpreter_wrapper${LIBRARY_EXTENSION}"

# Build python wheel.
cd "${BUILD_DIR}/tflite_pip"
case "${TENSORFLOW_TARGET}" in
  armhf)
    ${PYTHON} setup_with_binary.py bdist --plat-name=linux-armv7l \
                       bdist_wheel --plat-name=linux-armv7l
    ;;
  rpi0)
    ${PYTHON} setup_with_binary.py bdist --plat-name=linux_armv6l \
                       bdist_wheel --plat-name=linux-armv6l
    ;;
  aarch64)
    ${PYTHON} setup_with_binary.py bdist --plat-name=linux-aarch64 \
                       bdist_wheel --plat-name=linux-aarch64
    ;;
  *)
    if [[ -n "${TENSORFLOW_TARGET}" ]] && [[ -n "${TENSORFLOW_TARGET_ARCH}" ]]; then
      ${PYTHON} setup_with_binary.py bdist --plat-name=${TENSORFLOW_TARGET}-${TENSORFLOW_TARGET_ARCH} \
                         bdist_wheel --plat-name=${TENSORFLOW_TARGET}-${TENSORFLOW_TARGET_ARCH}
    else
      ${PYTHON} setup_with_binary.py bdist bdist_wheel
    fi
    ;;
esac

echo "Output can be found here:"
find "${BUILD_DIR}/tflite_pip/dist"

# Build debian package.
if [[ "${BUILD_DEB}" != "y" ]]; then
  exit 0
fi

PYTHON_VERSION=$(${PYTHON} -c "import sys;print(sys.version_info.major)")
if [[ ${PYTHON_VERSION} != 3 ]]; then
  echo "Debian package can only be generated for python3." >&2
  exit 1
fi

DEB_VERSION=$(dpkg-parsechangelog --show-field Version | cut -d- -f1)
if [[ "${DEB_VERSION}" != "${PACKAGE_VERSION}" ]]; then
  cat << EOF > "${BUILD_DIR}/tflite_pip/debian/changelog"
tflite-runtime (${PACKAGE_VERSION}-1) unstable; urgency=low

  * Bump version to ${PACKAGE_VERSION}.

 -- TensorFlow team <packages@tensorflow.org>  $(date -R)

$(<"${BUILD_DIR}/tflite_pip/debian/changelog")
EOF
fi

case "${TENSORFLOW_TARGET}" in
  armhf)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armhf
    ;;
  rpi0)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a armel
    ;;
  aarch64)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d -a arm64
    ;;
  *)
    dpkg-buildpackage -b -rfakeroot -us -uc -tc -d
    ;;
esac

cat "${BUILD_DIR}//tflite_pip/debian/changelog"

