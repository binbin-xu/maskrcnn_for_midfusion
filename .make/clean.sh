#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..
INSTALL_DIR=$ROOT/..

cd $INSTALL_DIR
rm -fr cocoapi
rm -fr apex

cd $ROOT
rm -fr build
rm -rf .anaconda3