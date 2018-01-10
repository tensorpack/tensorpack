import setuptools
version = int(setuptools.__version__.split('.')[0])
assert version > 30, "tensorpack installation requires setuptools > 30"
from setuptools import setup
import os
import shutil
import sys

# setup metainfo
CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, 'tensorpack/libinfo.py')
exec(open(libinfo_py, "rb").read())

# produce rst readme for pypi
try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except ImportError:
    long_description = open('README.md').read()

# configure requirements
reqfile = os.path.join(CURRENT_DIR, 'requirements.txt')
req = [x.strip() for x in open(reqfile).readlines()]

setup(
    name='tensorpack',
    version=__version__,
    description='Neural Network Toolbox on TensorFlow',
    long_description=long_description,

    install_requires=req,
    tests_require=['flake8', 'scikit-image'],
    extras_require={
        'all': ['pillow', 'scipy', 'h5py', 'lmdb>=0.92', 'matplotlib', 'scikit-learn'],
        'all: python_version < "3.0"': ['tornado']
    },

    #include_package_data=True,
    #package_data={'tensorpack': []},
)
