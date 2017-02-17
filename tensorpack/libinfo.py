
# issue#523 may happen on old systems
import cv2  # noqa

# issue#7378 may happen with custom opencv. It doesn't hurt to disable opencl
import os
os.environ['OPENCV_OPENCL_RUNTIME'] = ''

__version__ = '0.1.6'
