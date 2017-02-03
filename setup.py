from setuptools import setup
import sys

req = ['numpy',
       'six',
       'termcolor',
       'tqdm>4.11.1',
       'msgpack-python',
       'msgpack-numpy',
       'pyzmq'
       ]
if sys.version_info.major == 2:
    req.extend(['subprocess32', 'functools32'])

setup(version='0.1',
      install_requires=req,
      zip_safe=False    # dataset and __init__ use file
      )
