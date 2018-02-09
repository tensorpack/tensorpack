#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: nvml.py

from ctypes import (byref, c_uint, c_ulonglong,
                    CDLL, create_string_buffer,
                    POINTER, Structure)
import threading


__all__ = ['NvidiaContext']


_nvmlReturn_t = c_uint

NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_DEVICE_UUID_BUFFER_SIZE = 80
NVML_DEVICE_NAME_BUFFER_SIZE = 64


nvmlLib = None
_lib_lock = threading.Lock()

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


def NvmlCodeToString(i):
    return NvmlErrorCodes[str(i)]


class NvmlException(Exception):
    def __init__(self, error_code):
        super(NvmlException, self).__init__(error_code)
        self.error_code = error_code

    def __str__(self):
        return NvmlCodeToString(self.error_code)


def CheckNvmlReturn(ret):
    if (ret != 0):
        raise NvmlException(ret)
    return ret


class NVML(object):

    def load(self):
        global nvmlLib
        _lib_lock.acquire()
        try:
            nvmlLib = CDLL("libnvidia-ml.so.1")
        except OSError:
            NvmlException(NVML_ERROR_LIBRARY_NOT_FOUND)
        finally:
            _lib_lock.release()

        self.build_cache()

    def function_pointer(self, name):
        _lib_lock.acquire()
        try:
            try:
                return getattr(nvmlLib, name)
            except AttributeError:
                raise NvmlException(NVML_ERROR_FUNCTION_NOT_FOUND)
        finally:
            _lib_lock.release()

    def build_cache(self):
        function_pointers = ["nvmlDeviceGetName", "nvmlDeviceGetUUID", "nvmlDeviceGetMemoryInfo",
                             "nvmlDeviceGetUtilizationRates", "nvmlInit_v2", "nvmlShutdown",
                             "nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2"]

        self.func_ptr = {n: self.function_pointer(
            n) for n in function_pointers}

    def get_function(self, name):
        if name in self.func_ptr.keys():
            return self.func_ptr[name]

    def call(self, hnd, buf_len, func_ptr):
        c_name = create_string_buffer(buf_len)
        # fn = function_pointer_cache[func_ptr]
        fn = self.func_ptr[func_ptr]
        ret = fn(hnd, c_name, c_uint(buf_len))
        CheckNvmlReturn(ret)
        return c_name


_NVML = NVML()


class GpuDevice(Structure):
    """Represent GPU Information
    """
    pass


c_nvmlDevice_t = POINTER(GpuDevice)


class NvidiaDevice(object):
    """Represent a single GPUDevice"""

    def __init__(self, hnd):
        super(NvidiaDevice, self).__init__()
        self.hnd = hnd

    def Name(self):
        """Return GPU name

        Example:

            >>> print(nvidia.Device(0).Name())
            GeForce GTX 970

        Returns:
            Name of GPU brand
        """
        return _NVML.call(self.hnd, NVML_DEVICE_NAME_BUFFER_SIZE, "nvmlDeviceGetName").value

    def Memory(self):
        """Memory information in bytes

        Example:

            >>> print(nvidia.Device(0).Memory())
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
        CheckNvmlReturn(_NVML.get_function(
            "nvmlDeviceGetMemoryInfo")(self.hnd, byref(c_memory)))
        return {'total': c_memory.total, 'free': c_memory.free, 'used': c_memory.used}

    def Utilization(self):
        """Percent of time over the past second was utilized.

        Details:
           Percent of time over the past second during which one or more kernels was executing on the GPU.
           Percent of time over the past second during which global (device) memory was being read or written

        Example:

            >>> print(nvidia.Device(0).Memory())
            {'gpu': 4L, 'memory': 6L}

        """
        class GpuUtilizationInfo(Structure):

            _fields_ = [
                ('gpu', c_uint),
                ('memory', c_uint),
            ]

        c_util = GpuUtilizationInfo()
        CheckNvmlReturn(_NVML.get_function(
            "nvmlDeviceGetUtilizationRates")(self.hnd, byref(c_util)))
        return {'gpu': c_util.gpu, 'memory': c_util.memory}


class NvidiaContext(object):
    """Creates a context to query information

    Example:

        nvidia = NvidiaContext()
        nvidia.create_context()

        num_gpus = nvidia.NumCudaDevices()

        for device in nvidia.Devices():
            print(device.Name())

            print(device.Memory())
            print(device.Utilization())

        nvidia.destroy_context()
    """

    def __init__(self):
        super(NvidiaContext, self).__init__()

    def create_context(self):
        """Create a new context
        """
        _NVML.load()
        CheckNvmlReturn(_NVML.get_function("nvmlInit_v2")())

    def destroy_context(self):
        """Destroy current context
        """
        CheckNvmlReturn(_NVML.get_function("nvmlShutdown")())

    def NumCudaDevices(self):
        """Get number of CUDA devices.

        Example:
            >>> num_gpus = nvidia.NumCudaDevices()
            1

        Returns:
            count CUDA devices
        """
        c_count = c_uint()
        CheckNvmlReturn(_NVML.get_function(
            "nvmlDeviceGetCount_v2")(byref(c_count)))
        return c_count.value

    def Devices(self):
        """Generator of devices in context

        Yields:
            NvidiaDevice: single CUDA device
        """
        for i in range(self.NumCudaDevices()):
            yield self.Device(i)

    def Device(self, idx):
        """Get specific CUDA device

        Args:
            idx: index of device

        Returns:
            NvidiaDevice: single CUDA device
        """
        c_index = c_uint(idx)
        device = c_nvmlDevice_t()
        CheckNvmlReturn(_NVML.get_function(
            "nvmlDeviceGetHandleByIndex_v2")(c_index, byref(device)))
        return NvidiaDevice(device)
