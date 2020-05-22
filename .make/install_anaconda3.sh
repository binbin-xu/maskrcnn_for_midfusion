#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

if [ ! -d $ROOT/.anaconda3 ]; then
  echo "==>Installing anaconda 3"
  cd $ROOT
  curl -L https://www.doc.ic.ac.uk/~bx516/tools/install_anaconda3.sh | bash -s .
fi

