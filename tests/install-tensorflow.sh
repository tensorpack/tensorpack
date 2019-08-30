#!/bin/bash -e

if [ $TF_VERSION == "nightly" ]; then
	TF_BINARY_URL="tf-nightly"
else
  if [[ $TRAVIS_PYTHON_VERSION == 2* ]]; then
		TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TF_VERSION}-cp27-none-linux_x86_64.whl
	fi
	if [[ $TRAVIS_PYTHON_VERSION == 3.4* ]]; then
		TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TF_VERSION}-cp34-cp34m-linux_x86_64.whl
	fi
	if [[ $TRAVIS_PYTHON_VERSION == 3.5* ]]; then
		TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TF_VERSION}-cp35-cp35m-linux_x86_64.whl
	fi
	if [[ $TRAVIS_PYTHON_VERSION == 3.6* ]]; then
		TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TF_VERSION}-cp36-cp36m-linux_x86_64.whl
	fi
fi


python -m pip install --upgrade ${TF_BINARY_URL}
