#!/bin/bash

set -e

echo_bold () {
  echo -e "\033[1m$*\033[0m"
}

echo_warning () {
  echo -e "\033[33m$*\033[0m"
}

conda_check_installed () {
  if [ ! $# -eq 1 ]; then
    echo "usage: $0 PACKAGE_NAME"
    return 1
  fi
  conda list | awk '{print $1}' | egrep "^$1$" &>/dev/null
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

cd $ROOT

source .anaconda3/bin/activate

# ---------------------------------------------------------------------------------------

echo_bold "==> Installing the right pip and dependencies for the fresh python"
# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# maskrcnn_benchmark and coco api dependencies
yes | pip install ninja yacs cython matplotlib tqdm opencv-python

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
# conda install -c pytorch pytorch-nightly torchvision cudatoolkit=9.0
# modified for CUDA 10.0
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch

echo_bold "==> Installing dependencies"

export INSTALL_DIR=$PWD/..

# install pycocotools
if test -d "$INSTALL_DIR/cocoapi"; then
  rm -fr $INSTALL_DIR/cocoapi
fi
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
if test -d "$INSTALL_DIR/apex"; then
  rm -fr $INSTALL_DIR/apex
fi
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
echo_bold "==> Installing PyTorch Detection"
cd $INSTALL_DIR/maskrcnn_for_midfusion

# # the following will install the lib with
# # symbolic links, so that you can modify
# # the files if you want and won't need to
# # re-build it
python setup.py build develop

unset INSTALL_DIR

echo_bold "\nAll is well! You can start using this!
  $ source .anaconda3/bin/activate
"
