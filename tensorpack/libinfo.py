
import os

# issue#7378 may happen with custom opencv. It doesn't hurt to disable opencl
os.environ['OPENCV_OPENCL_RUNTIME'] = ''
try:
    # issue#1924 may happen on old systems
    import cv2  # noqa
    if int(cv2.__version__.split('.')[0]) == 3:
        cv2.ocl.setUseOpenCL(False)
except ImportError:
    pass

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # issue#9339
os.environ['TF_AUTOTUNE_THRESHOLD'] = '3'   # use more warm-up
os.environ['TF_AVGPOOL_USE_CUDNN'] = '1'   # issue#8566

__version__ = '0.3.0'
