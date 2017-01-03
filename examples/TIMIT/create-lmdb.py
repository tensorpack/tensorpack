#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: create-lmdb.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import sys
import os
import scipy.io.wavfile as wavfile
import string
import numpy as np
import argparse

from tensorpack import *
from tensorpack.utils.argtools import memoized
from tensorpack.utils.stats import OnlineMoments
import bob.ap

CHARSET = set(string.ascii_lowercase + ' ')
PHONEME_LIST = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
    'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r',
    's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

PHONEME_DIC = {v: k for k, v in enumerate(PHONEME_LIST)}
WORD_DIC = {v: k for k, v in enumerate(string.ascii_lowercase + ' ')}


def read_timit_txt(f):
    f = open(f)
    line = f.readlines()[0].strip().split(' ')
    line = line[2:]
    line = ' '.join(line)
    line = line.replace('.', '').lower()
    line = filter(lambda c: c in CHARSET, line)
    f.close()
    for c in line:
        ret.append(WORD_DIC[c])
    return np.asarray(ret)


def read_timit_phoneme(f):
    f = open(f)
    pho = []
    for line in f:
        line = line.strip().split(' ')[-1]
        pho.append(PHONEME_DIC[line])
    f.close()
    return np.asarray(pho)


@memoized
def get_bob_extractor(fs, win_length_ms=10, win_shift_ms=5,
                      n_filters=55, n_ceps=15, f_min=0., f_max=6000,
                      delta_win=2, pre_emphasis_coef=0.95, dct_norm=True,
                      mel_scale=True):
    ret = bob.ap.Ceps(fs, win_length_ms, win_shift_ms, n_filters, n_ceps, f_min,
                      f_max, delta_win, pre_emphasis_coef, mel_scale, dct_norm)
    return ret


def diff_feature(feat, nd=1):
    diff = feat[1:] - feat[:-1]
    feat = feat[1:]
    if nd == 1:
        return np.concatenate((feat, diff), axis=1)
    elif nd == 2:
        d2 = diff[1:] - diff[:-1]
        return np.concatenate((feat[1:], diff[1:], d2), axis=1)


def get_feature(f):
    fs, signal = wavfile.read(f)
    signal = signal.astype('float64')
    feat = get_bob_extractor(fs, n_filters=26, n_ceps=13)(signal)
    feat = diff_feature(feat, nd=2)
    return feat


class RawTIMIT(DataFlow):

    def __init__(self, dirname, label='phoneme'):
        self.dirname = dirname
        assert os.path.isdir(dirname), dirname
        self.filelists = [k for k in fs.recursive_walk(self.dirname)
                          if k.endswith('.wav')]
        logger.info("Found {} wav files ...".format(len(self.filelists)))
        assert len(self.filelists), self.filelists
        assert label in ['phoneme', 'letter'], label
        self.label = label

    def size(self):
        return len(self.filelists)

    def get_data(self):
        for f in self.filelists:
            feat = get_feature(f)
            if self.label == 'phoneme':
                label = read_timit_phoneme(f[:-4] + '.PHN')
            elif self.label == 'letter':
                label = read_timit_txt(f[:-4] + '.TXT')
            yield [feat, label]


def compute_mean_std(db, fname):
    ds = LMDBDataPoint(db, shuffle=False)
    o = OnlineMoments()
    with get_tqdm(total=ds.size()) as bar:
        for dp in ds.get_data():
            feat = dp[0]  # len x dim
            for f in feat:
                o.feed(f)
            bar.update()
    logger.info("Writing to {} ...".format(fname))
    with open(fname, 'wb') as f:
        f.write(serialize.dumps([o.mean, o.std]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='command', dest='command')
    parser_db = subparsers.add_parser('build', help='build a LMDB database')
    parser_db.add_argument('--dataset',
                           help='path to TIMIT TRAIN or TEST directory', required=True)
    parser_db.add_argument('--db', help='output lmdb file', required=True)

    parser_stat = subparsers.add_parser('stat', help='compute statistics (mean/std) of dataset')
    parser_stat.add_argument('--db', help='input lmdb file', required=True)
    parser_stat.add_argument('-o', '--output',
                             help='output statistics file', default='stats.data')

    args = parser.parse_args()
    if args.command == 'build':
        ds = RawTIMIT(args.dataset)
        dftools.dump_dataflow_to_lmdb(ds, args.db)
    elif args.command == 'stat':
        compute_mean_std(args.db, args.output)
