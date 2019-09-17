# -*- coding: utf-8 -*-
# File: nvml.py

import threading
from ctypes import (
    CDLL, POINTER, Structure, byref, c_uint,
    c_ulonglong, create_string_buffer)

__all__ = ['NVMLContext']


NVML_ERROR_FUNCTION_NOT_FOUND = 13


NvmlErrorCodes = {"0": "NVML_SUCCESS",
                  "1": "NVML_ERROR_UNINITIALIZED",
                  "2": "NVML_ERROR_INVALID_ARGUMENT",
                  "3": "NVML_ERROR_NOT_SUPPORTED",
                  "4": "NVML_ERROR_NO_PERMISSION",
                  "5": "NVML_ERROR_ALREADY_INITIALIZED",
                  "6": "NVML_ERROR_NOT_FOUND",
                  "7": "NVML_ERROR_INSUFFICIENT_SIZE",
                  "8": "NVML_ERROR_INSUFFICIENT_POWER",
                  "9": "NVML_ERROR_DRIVER_NOT_LOADED",
                  "10": "NVML_ERROR_TIMEOUT",
                  "11": "NVML_ERROR_IRQ_ISSUE",
                  "12": "NVML_ERROR_LIBRARY_NOT_FOUND",
                  "13": "NVML_ERROR_FUNCTION_NOT_FOUND",
                  "14": "NVML_ERROR_CORRUPTED_INFOROM",
                  "15": "NVML_ERROR_GPU_IS_LOST",
                  "16": "NVML_ERROR_RESET_REQUIRED",
                  "17": "NVML_ERROR_OPERATING_SYSTEM",
                  "18": "NVML_ERROR_LIB_RM_VERSION_MISMATCH",
                  "999": "NVML_ERROR_UNKNOWN"}


class NvmlException(Exception):
    def __init__(self, error_code):
        super(NvmlException, self).__init__(error_code)
        self.error_code = error_code

    def __str__(self):
        return NvmlErrorCodes[str(self.error_code)]


def _check_return(ret):
    if (ret != 0):
        raise NvmlException(ret)
    return ret


class NVML(object):
    """
    Loader for libnvidia-ml.so
    """

    _nvmlLib = None
    _lib_lock = threading.Lock()

    def load(self):
        with self._lib_lock:
            if self._nvmlLib is None:
                self._nvmlLib = CDLL("libnvidia-ml.so.1")

                function_pointers = ["nvmlDeviceGetName", "nvmlDeviceGetUUID", "nvmlDeviceGetMemoryInfo",
                                     "nvmlDeviceGetUtilizationRates", "nvmlInit_v2", "nvmlShutdown",
                                     "nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2"]

                self.func_ptr = {n: self._function_pointer(n) for n in function_pointers}

    def _function_pointer(self, name):
        try:
            return getattr(self._nvmlLib, name)
        except AttributeError:
            raise NvmlException(NVML_ERROR_FUNCTION_NOT_FOUND)

    def get_function(self, name):
        if name in self.func_ptr.keys():
            return self.func_ptr[name]


_NVML = NVML()


class NvidiaDevice(object):
    """Represent a single GPUDevice"""

    def __init__(self, hnd):
        super(NvidiaDevice, self).__init__()
        self.hnd = hnd

    def memory(self):
        """Memory information in bytes

        Example:

            >>> print(ctx.device(0).memory())
            {'total': 4238016512L, 'used': 434831360L, 'free': 3803185152L}

        Returns:
            total/used/free memory in bytes
        """
        class GpuMemoryInfo(Structure):
            _fields_ = [
                ('total', c_ulonglong),
                ('free', c_ulonglong),
                ('used', c_ulonglong),
            ]

        c_memory = GpuMemoryInfo()
        _check_return(_NVML.get_function(
            "nvmlDeviceGetMemoryInfo")(self.hnd, byref(c_memory)))
        return {'total': c_memory.total, 'free': c_memory.free, 'used': c_memory.used}

    def utilization(self):
        """Percent of time over the past second was utilized.

        Details:
           Percent of time over the past second during which one or more kernels was executing on the GPU.
           Percent of time over the past second during which global (device) memory was being read or written

        Example:

            >>> print(ctx.device(0).utilization())
            {'gpu': 4L, 'memory': 6L}

        """
        class GpuUtilizationInfo(Structure):

            _fields_ = [
                ('gpu', c_uint),
                ('memory', c_uint),
            ]

        c_util = GpuUtilizationInfo()
        _check_return(_NVML.get_function(
            "nvmlDeviceGetUtilizationRates")(self.hnd, byref(c_util)))
        return {'gpu': c_util.gpu, 'memory': c_util.memory}

    def name(self):
        buflen = 1024
        buf = create_string_buffer(buflen)
        fn = _NVML.get_function("nvmlDeviceGetName")
        ret = fn(self.hnd, buf, c_uint(1024))
        _check_return(ret)
        return buf.value.decode('utf-8')


class NVMLContext(object):
    """Creates a context to query information

    Example:

        with NVMLContext() as ctx:
            num_gpus = ctx.num_devices()
            for device in ctx.devices():
                print(device.memory())
                print(device.utilization())

    """
    def __enter__(self):
        """Create a new context """
        _NVML.load()
        _check_return(_NVML.get_function("nvmlInit_v2")())
        return self

    def __exit__(self, type, value, tb):
        """Destroy current context"""
        _check_return(_NVML.get_function("nvmlShutdown")())

    def num_devices(self):
        """Get number of devices """
        c_count = c_uint()
        _check_return(_NVML.get_function(
            "nvmlDeviceGetCount_v2")(byref(c_count)))
        return c_count.value

    def devices(self):
        """
        Returns:
            [NvidiaDevice]: a list of devices
        """
        return [self.device(i) for i in range(self.num_devices())]

    def device(self, idx):
        """Get a specific GPU device

        Args:
            idx: index of device

        Returns:
            NvidiaDevice: single GPU device
        """

        class GpuDevice(Structure):
            pass

        c_nvmlDevice_t = POINTER(GpuDevice)

        c_index = c_uint(idx)
        device = c_nvmlDevice_t()
        _check_return(_NVML.get_function(
            "nvmlDeviceGetHandleByIndex_v2")(c_index, byref(device)))
        return NvidiaDevice(device)


if __name__ == '__main__':
    with NVMLContext() as ctx:
        for idx, dev in enumerate(ctx.devices()):
            print(idx, dev.name())

    with NVMLContext() as ctx:
        print(ctx.devices())
        print(ctx.devices()[0].utilization())
