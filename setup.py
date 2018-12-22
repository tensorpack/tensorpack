import platform
from os import path
import setuptools
from setuptools import setup

version = int(setuptools.__version__.split('.')[0])
assert version > 30, "Tensorpack installation requires setuptools > 30"

this_directory = path.abspath(path.dirname(__file__))

# setup metainfo
libinfo_py = path.join(this_directory, 'tensorpack', 'libinfo.py')
libinfo_content = open(libinfo_py, "r").readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

with open(path.join(this_directory, 'README.md'), 'rb') as f:
    long_description = f.read().decode('utf-8')


def add_git_version():

    def get_git_version():
        from subprocess import check_output
        try:
            return check_output("git describe --tags --long --dirty".split()).decode('utf-8').strip()
        except Exception:
            return __version__  # noqa

    newlibinfo_content = [l for l in libinfo_content if not l.startswith('__git_version__')]
    newlibinfo_content.append('__git_version__ = "{}"'.format(get_git_version()))
    with open(libinfo_py, "w") as f:
        f.write("".join(newlibinfo_content))


add_git_version()


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
        "msgpack-numpy>=0.4.4.2",
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
