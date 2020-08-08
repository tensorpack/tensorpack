
import os
import sys
import psutil
import tensorflow as tf
import numpy as np
from collections import defaultdict, OrderedDict
from tabulate import tabulate

import tensorpack
from ..compat import tfv1
from ..utils.utils import find_library_full_path as find_library
from ..utils.nvml import NVMLContext
from ..libinfo import __git_version__


def parse_TF_build_info():
    ret = OrderedDict()
    from tensorflow.python.platform import build_info
    try:
        for k, v in list(build_info.build_info.items()):
            if k == "cuda_version":
                ret["TF built with CUDA"] = v
            elif k == "cudnn_version":
                ret["TF built with CUDNN"] = v
            elif k == "cuda_compute_capabilities":
                ret["TF compute capabilities"] = ",".join([k.replace("compute_", "") for k in v])
        return ret
    except AttributeError:
        pass
    try:
        ret["TF built with CUDA"] = build_info.cuda_version_number
        ret["TF built with CUDNN"] = build_info.cudnn_version_number
    except AttributeError:
        pass
    return ret


def collect_env_info():
    """
    Returns:
        str - a table contains important information about the environment
    """
    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("Tensorpack", __git_version__ + " @" + os.path.dirname(tensorpack.__file__)))
    data.append(("Numpy", np.__version__))

    data.append(("TensorFlow", tfv1.VERSION + "/" + tfv1.GIT_VERSION + " @" + os.path.dirname(tf.__file__)))
    data.append(("TF Compiler Version", tfv1.COMPILER_VERSION))
    has_cuda = tf.test.is_built_with_cuda()
    data.append(("TF CUDA support", has_cuda))

    try:
        from tensorflow.python.framework import test_util
        data.append(("TF MKL support", test_util.IsMklEnabled()))
    except Exception:
        pass

    try:
        from tensorflow.python.framework import test_util
        data.append(("TF XLA support", test_util.is_xla_enabled()))
    except Exception:
        pass

    if has_cuda:
        data.append(("Nvidia Driver", find_library("nvidia-ml")))
        data.append(("CUDA libs", find_library("cudart")))
        data.append(("CUDNN libs", find_library("cudnn")))
        for k, v in parse_TF_build_info().items():
            data.append((k, v))
        data.append(("NCCL libs", find_library("nccl")))

        # List devices with NVML
        data.append(
            ("CUDA_VISIBLE_DEVICES",
             os.environ.get("CUDA_VISIBLE_DEVICES", "Unspecified")))
        try:
            devs = defaultdict(list)
            with NVMLContext() as ctx:
                for idx, dev in enumerate(ctx.devices()):
                    devs[dev.name()].append(str(idx))

            for devname, devids in devs.items():
                data.append(
                    ("GPU " + ",".join(devids), devname))
        except Exception:
            data.append(("GPU", "Not found with NVML"))

    vram = psutil.virtual_memory()
    data.append(("Free RAM", "{:.2f}/{:.2f} GB".format(vram.available / 1024**3, vram.total / 1024**3)))
    data.append(("CPU Count", psutil.cpu_count()))

    # Other important dependencies:
    try:
        import horovod
        data.append(("Horovod", horovod.__version__ + " @" + os.path.dirname(horovod.__file__)))
    except ImportError:
        pass

    try:
        import cv2
        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass

    import msgpack
    data.append(("msgpack", ".".join([str(x) for x in msgpack.version])))

    has_prctl = True
    try:
        import prctl
        _ = prctl.set_pdeathsig  # noqa
    except Exception:
        has_prctl = False
    data.append(("python-prctl", has_prctl))

    return tabulate(data)


if __name__ == '__main__':
    print(collect_env_info())
    print("Detecting GPUs using TensorFlow:")
    try:
        # available since TF 1.14
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        gpu_devices = [x.name for x in gpu_devices]
    except AttributeError:
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        gpu_devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
    print("GPUs:", ", ".join(gpu_devices))
