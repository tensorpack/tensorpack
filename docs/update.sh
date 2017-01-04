#!/bin/bash -e
# File: update.sh
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


PROG_NAME=`readlink -f $0`
PROG_DIR=`dirname "$PROG_NAME"`
cd "$PROG_DIR"

make clean
#sphinx-apidoc -o modules ../tensorpack -f -d 10
make html
