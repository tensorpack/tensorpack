#!/bin/bash

set -v

. /workspace/venv/bin/activate

pip install -e .
pip install tensorflow-gpu==1.15.0
pip install opencv-python-headless scipy pycocotools>=2.0.1

echo "Extracting ..."
cd ./.data/vision/coco
python -c 'import zipfile; zipfile.ZipFile("annotations_trainval2017.zip").extractall()'
python -c 'import zipfile; zipfile.ZipFile("val2017.zip").extractall()'
cd -
