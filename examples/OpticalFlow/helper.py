#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper for Optical Flow visualization
"""

import numpy as np


class Flow(object):
    """
    based on https://github.com/cgtuebingen/learning-blind-motion-deblurring/blob/master/synthblur/src/flow.cpp#L44
    """
    def __init__(self):
        super(Flow, self).__init__()
        self.wheel = None
        self._construct_wheel()

    @staticmethod
    def read(file):
        # https://stackoverflow.com/a/44906777/7443104
        with open(file, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                raise Exception('Magic number incorrect. Invalid .flo file')
            else:
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]
                data = np.fromfile(f, np.float32, count=2 * w * h)
                return np.resize(data, (h, w, 2))

    def _construct_wheel(self):
        k = 0

        RY, YG, GC = 15, 6, 4
        YG, GC, CB = 6, 4, 11
        BM, MR = 13, 6

        self.wheel = np.zeros((55, 3), dtype=np.float32)

        for i in range(RY):
            self.wheel[k] = np.array([255., 255. * i / float(RY), 0])
            k += 1

        for i in range(YG):
            self.wheel[k] = np.array([255. - 255. * i / float(YG), 255., 0])
            k += 1

        for i in range(GC):
            self.wheel[k] = np.array([0, 255., 255. * i / float(GC)])
            k += 1

        for i in range(CB):
            self.wheel[k] = np.array([0, 255. - 255. * i / float(CB), 255.])
            k += 1

        for i in range(BM):
            self.wheel[k] = np.array([255. * i / float(BM), 0, 255.])
            k += 1

        for i in range(MR):
            self.wheel[k] = np.array([255., 0, 255. - 255. * i / float(MR)])
            k += 1

        self.wheel = self.wheel / 255.

    def visualize(self, nnf):
        assert len(nnf.shape) == 3
        assert nnf.shape[2] == 2

        RY, YG, GC = 15, 6, 4
        YG, GC, CB = 6, 4, 11
        BM, MR = 13, 6
        NCOLS = RY + YG + GC + CB + BM + MR

        fx = nnf[:, :, 0].astype(np.float32)
        fy = nnf[:, :, 1].astype(np.float32)

        h, w = fx.shape[:2]
        fx = fx.reshape([-1])
        fy = fy.reshape([-1])

        rad = np.sqrt(fx * fx + fy * fy)

        max_rad = rad.max()

        a = np.arctan2(-fy, -fx) / np.pi
        fk = (a + 1.0) / 2.0 * (NCOLS - 1)
        k0 = fk.astype(np.int32)
        k1 = (k0 + 1) % NCOLS
        f = (fk - k0).astype(np.float32)

        color0 = self.wheel[k0, :]
        color1 = self.wheel[k1, :]

        f = np.stack([f, f, f], axis=-1)
        color = (1 - f) * color0 + f * color1

        color = 1 - (np.expand_dims(rad, axis=-1) / max_rad) * (1 - color)

        return color.reshape(h, w, 3)[:, :, ::-1]


if __name__ == '__main__':
    import cv2
    nnf = Flow.read('/tmp/data2/07446_flow.flo')
    v = Flow()
    rgb = v.visualize(nnf)
    cv2.imshow('rgb', rgb)
    cv2.waitKey(0)
