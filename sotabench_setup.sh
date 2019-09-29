#!/bin/bash

. /workspace/venv/bin/activate

pip install -e .

cd ./.data/vision/coco
python -c 'import zipfile; zipfile.ZipFile("annotations_trainval2017.zip").extractall()'
python -c 'import zipfile; zipfile.ZipFile("val2017.zip").extractall()'
cd -
