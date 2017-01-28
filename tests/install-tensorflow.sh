#!/bin/bash -e

if [ $TF_TYPE == "release" ]; then
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
if [ $TF_TYPE == "nightly" ]; then
	if [[ $TRAVIS_PYTHON_VERSION == 2* ]]; then
		TF_BINARY_URL=https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-${TF_VERSION}-cp27-none-linux_x86_64.whl

	fi
	if [[ $TRAVIS_PYTHON_VERSION == 3.4* ]]; then
		TF_BINARY_URL=https://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-${TF_VERSION}-cp34-cp34m-linux_x86_64.whl
	fi
	if [[ $TRAVIS_PYTHON_VERSION == 3.5* ]]; then
		TF_BINARY_URL=https://ci.tensorflow.org/view/Nightly/job/nightly-python35-linux-cpu/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-${TF_VERSION}-cp35-cp35m-linux_x86_64.whl
	fi
fi


pip install  --upgrade ${TF_BINARY_URL}
