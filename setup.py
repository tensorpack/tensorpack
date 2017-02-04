from setuptools import setup

req = [
    'numpy',
    'six',
    'termcolor',
    'tqdm>4.11.1',
    'msgpack-python',
    'msgpack-numpy',
    'pyzmq',
    'subprocess32;python_version<"3.0"',
    'functools32;python_version<"3.0"',
]
extra_req = [
    'pillow',
    'scipy',
    'h5py',
    'lmdb',
    'matplotlib',
    'scikit-learn',
    'tornado;python_version<"3.0"',
]

# TODO:
# setup_requires, scripts
setup(
    install_requires=req,
    extras_require={
        'all': extra_req
    },
)
