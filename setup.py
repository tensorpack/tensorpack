import setuptools
from setuptools import setup
from os import path
import platform

version = int(setuptools.__version__.split('.')[0])
assert version > 30, "Tensorpack installation requires setuptools > 30"

this_directory = path.abspath(path.dirname(__file__))

# setup metainfo
libinfo_py = path.join(this_directory, 'tensorpack', 'libinfo.py')
last_line = open(libinfo_py, "rb").readlines()[-1].strip()
exec(last_line)

with open(path.join(this_directory, 'README.md'), 'rb') as f:
    long_description = f.read().decode('utf-8')

setup(
    name='tensorpack',
    version=__version__,   # noqa
    description='Neural Network Toolbox on TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "numpy>=1.14",
        "six",
        "termcolor>=1.1",
        "tabulate>=0.7.7",
        "tqdm>4.11.1",
        "msgpack>=0.5.2",
        "msgpack-numpy>=0.4.0",
        "pyzmq>=16",
        "subprocess32; python_version < '3.0'",
        "functools32; python_version < '3.0'",
    ],
    tests_require=['flake8', 'scikit-image'],
    extras_require={
        'all': ['pillow', 'scipy', 'h5py', 'lmdb>=0.92', 'matplotlib', 'scikit-learn'] +
               ['python-prctl'] if platform.system() == 'Linux' else [],
        'all: python_version < "3.0"': ['tornado'],
    },
)
