# -*- coding: utf-8 -*-
# File: timitdata.py
# Author: Yuxin Wu

import numpy as np
from six.moves import range

from tensorpack import ProxyDataFlow

__all__ = ['TIMITBatch']


def batch_feature(feats):
    # pad to the longest in the batch
    maxlen = max([k.shape[0] for k in feats])
    bsize = len(feats)
    ret = np.zeros((bsize, maxlen, feats[0].shape[1]))
    for idx, feat in enumerate(feats):
        ret[idx, :feat.shape[0], :] = feat
    return ret


def sparse_label(labels):
    maxlen = max([k.shape[0] for k in labels])
    shape = [len(labels), maxlen]   # bxt
    indices = []
    values = []
    for bid, lab in enumerate(labels):
        for tid, c in enumerate(lab):
            indices.append([bid, tid])
            values.append(c)
    indices = np.asarray(indices)
    values = np.asarray(values)
    return (indices, values, shape)


class TIMITBatch(ProxyDataFlow):

    def __init__(self, ds, batch):
        self.batch = batch
        self.ds = ds

    def __len__(self):
        return len(self.ds) // self.batch

    def __iter__(self):
        itr = self.ds.__iter__()
        for _ in range(self.__len__()):
            feats = []
            labs = []
            for b in range(self.batch):
                feat, lab = next(itr)
                feats.append(feat)
                labs.append(lab)
            batchfeat = batch_feature(feats)
            batchlab = sparse_label(labs)
            seqlen = np.asarray([k.shape[0] for k in feats])
            yield [batchfeat, batchlab[0], batchlab[1], batchlab[2], seqlen]
