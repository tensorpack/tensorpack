#!/bin/bash

set -v

. /workspace/venv/bin/activate

pip install -e .
pip install tensorflow-gpu==1.14.0
pip install opencv-python scipy

echo "Extracting ..."
cd ./.data/vision/coco
python -c 'import zipfile; zipfile.ZipFile("annotations_trainval2017.zip").extractall()'
python -c 'import zipfile; zipfile.ZipFile("val2017.zip").extractall()'
cd -
