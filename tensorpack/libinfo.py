
# issue#1924 may happen on old systems
import cv2  # noqa
import os

# issue#7378 may happen with custom opencv. It doesn't hurt to disable opencl
os.environ['OPENCV_OPENCL_RUNTIME'] = ''

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # issue#9339
os.environ['TF_AUTOTUNE_THRESHOLD'] = '3'   # use more warm-up
os.environ['TF_AVGPOOL_USE_CUDNN'] = '1'   # issue#8566

__version__ = '0.1.9'
