#!/usr/bin/env python3

import numpy as np
import argparse
import pyarrow as pa
from tabulate import tabulate
import operator
from tensorpack.utils import logger
from tensorpack.utils.serialize import (
    MsgpackSerializer,
    PyarrowSerializer,
    PickleSerializer,
    ForkingPickler,
)
from tensorpack.utils.timer import Timer


def benchmark_serializer(dumps, loads, data, num):
    buf = dumps(data)

    enc_timer = Timer()
    dec_timer = Timer()
    enc_timer.pause()
    dec_timer.pause()

    for k in range(num):
        enc_timer.resume()
        buf = dumps(data)
        enc_timer.pause()

        dec_timer.resume()
        loads(buf)
        dec_timer.pause()

    dumps_time = enc_timer.seconds() / num
    loads_time = dec_timer.seconds() / num
    return dumps_time, loads_time


def display_results(name, results):
    logger.info("Encoding benchmark for {}:".format(name))
    data = sorted(((x, y[0]) for x, y in results), key=operator.itemgetter(1))
    print(tabulate(data, floatfmt='.5f'))

    logger.info("Decoding benchmark for {}:".format(name))
    data = sorted(((x, y[1]) for x, y in results), key=operator.itemgetter(1))
    print(tabulate(data, floatfmt='.5f'))


def benchmark_all(name, serializers, data, num=30):
    logger.info("Benchmarking {} ...".format(name))
    results = []
    for serializer_name, dumps, loads in serializers:
        results.append((serializer_name, benchmark_serializer(dumps, loads, data, num=num)))
    display_results(name, results)


def fake_json_data():
    return {
        'words': """
            Lorem ipsum dolor sit amet, consectetur adipiscing
            elit. Mauris adipiscing adipiscing placerat.
            Vestibulum augue augue,
            pellentesque quis sollicitudin id, adipiscing.
            """ * 100,
        'list': list(range(100)) * 500,
        'dict': {str(i): 'a' for i in range(50000)},
        'dict2': {i: 'a' for i in range(50000)},
        'int': 3000,
        'float': 100.123456
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    args = parser.parse_args()

    serializers = [
        ("msgpack", MsgpackSerializer.dumps, MsgpackSerializer.loads),
        ("pyarrow-buf", PyarrowSerializer.dumps, PyarrowSerializer.loads),
        ("pyarrow-bytes", PyarrowSerializer.dumps_bytes, PyarrowSerializer.loads),
        ("pickle", PickleSerializer.dumps, PickleSerializer.loads),
        ("forking-pickle", ForkingPickler.dumps, ForkingPickler.loads),
    ]

    if args.task == "numpy":
        numpy_data = [np.random.rand(64, 224, 224, 3).astype("float32"), np.random.rand(64).astype('int32')]
        benchmark_all("numpy data", serializers, numpy_data)
    elif args.task == "json":
        benchmark_all("json data", serializers, fake_json_data(), num=50)
    elif args.task == "torch":
        import torch
        from pyarrow.lib import _default_serialization_context

        pa.register_torch_serialization_handlers(_default_serialization_context)
        torch_data = [torch.rand(64, 224, 224, 3), torch.rand(64).to(dtype=torch.int32)]
        benchmark_all("torch data", serializers[1:], torch_data)
