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
reqfile = os.path.join(CURRENT_DIR, 'opt-requirements.txt')
extra_req = [x.strip() for x in open(reqfile).readlines()]
if sys.version_info.major < 3:
    extra_req.append('tornado')

# parse scripts
scripts = ['scripts/plot-point.py', 'scripts/dump-model-params.py']
scripts_to_install = []
for s in scripts:
    dirname = os.path.dirname(s)
    basename = os.path.basename(s)
    if basename.endswith('.py'):
        basename = basename[:-3]
    newname = 'tpk-' + basename  # install scripts with a prefix to avoid name confusion
    # setup.py could be executed the second time in the same dir
    if not os.path.isfile(newname):
        shutil.move(s, newname)
    scripts_to_install.append(newname)

setup(
    name='tensorpack',
    version=__version__,
    description='Neural Network Toolbox on TensorFlow',
    long_description=long_description,
    install_requires=req,
    tests_require=['flake8', 'scikit-image'],
    extras_require={
        'all': extra_req
    },
    scripts=scripts_to_install,
)
