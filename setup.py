from os import path
import setuptools
from setuptools import setup, find_packages

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
    author="TensorPack contributors",
    author_email="ppwwyyxxc@gmail.com",
    url="https://github.com/tensorpack/tensorpack",
    keywords="tensorflow, deep learning, neural network",
    license="Apache",

    version=__version__,   # noqa
    description='A Neural Network Training Interface on TensorFlow',
    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=find_packages(exclude=["examples", "tests"]),
    zip_safe=False,  		    # dataset and __init__ use file

    install_requires=[
        "numpy>=1.14",
        "six",
        "termcolor>=1.1",
        "tabulate>=0.7.7",
        "tqdm>4.29.0",
        "msgpack>=0.5.2",
        "msgpack-numpy>=0.4.4.2",
        "pyzmq>=16",
        "psutil>=5",
    ],
    tests_require=['flake8', 'scikit-image'],
    extras_require={
        'all': ['scipy', 'h5py', 'lmdb>=0.92', 'matplotlib', 'scikit-learn'],
        'all: "linux" in sys_platform': ['python-prctl'],
    },

    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#universal-wheels
    options={'bdist_wheel': {'universal': '1'}},
)
