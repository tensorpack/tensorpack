
import os

# issue#7378 may happen with custom opencv. It doesn't hurt to disable opencl
os.environ['OPENCV_OPENCL_RUNTIME'] = 'disabled'     # https://github.com/opencv/opencv/pull/10155
try:
    # issue#1924 may happen on old systems
    import cv2  # noqa
    if int(cv2.__version__.split('.')[0]) == 3:
        cv2.ocl.setUseOpenCL(False)
    # check if cv is built with cuda or openmp
    info = cv2.getBuildInformation().split('\n')
    for line in info:
        splits = line.split()
        if not len(splits):
            continue
        answer = splits[-1].lower()
        if answer in ['yes', 'no']:
            if 'cuda' in line.lower() and answer == 'yes':
                # issue#1197
                print("OpenCV is built with CUDA support. "
                      "This may cause slow initialization or sometimes segfault with TensorFlow.")
        if answer == 'openmp':
            print("OpenCV is built with OpenMP support. This usually results in poor performance. For details, see "
                  "https://github.com/tensorpack/benchmarks/blob/master/ImageNet/benchmark-opencv-resize.py")
except (ImportError, TypeError):
    pass

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # issue#9339
os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'   # use more warm-up

# Since 1.3, this is not needed
os.environ['TF_AVGPOOL_USE_CUDNN'] = '1'   # issue#8566

# TF1.5 features
os.environ['TF_SYNC_ON_FINISH'] = '0'   # will become default
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'

# Available in TF1.6+. Haven't seen different performance on R50.
# NOTE TF set it to 0 by default, because:
# this mode may use scaled atomic integer reduction that may cause a numerical
# overflow for certain input data range.
# os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

try:
    import tensorflow as tf  # noqa
    _version = tf.__version__.split('.')
    assert int(_version[0]) >= 1 and int(_version[1]) >= 3, "TF>=1.3 is required!"
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


# This line has to be the last line of the file.
# setup.py will use it to determine the version
__version__ = '0.8.6'
