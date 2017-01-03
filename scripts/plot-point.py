#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A general curve plotter to create curves such as:
https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet

A simplest example:
$ cat examples/train_log/mnist-convnet/stat.json \
        | jq '.[] | .train_error, .validation_error' \
        | paste - - \
        | plot-point.py --legend 'train,val' --xlabel 'epoch' --ylabel 'error'

For more usage, see `plot-point.py -h` or the code.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fontm
import argparse
import sys
from collections import defaultdict
from itertools import chain
import six

# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'sans-serif','sans-serif':['Microsoft Yahei']})
# rc('text', usetex=True)

STDIN_FNAME = '-'


def get_args():
    description = "plot points into graph."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input',
                        help='input data file, use "-" for stdin. Default stdin. Input \
            format is many rows of DELIMIETER-separated data',
                        default='-')
    parser.add_argument('-o', '--output',
                        help='output image', default='')
    parser.add_argument('--show',
                        help='show the figure after rendered',
                        action='store_true')
    parser.add_argument('-c', '--column',
                        help="describe each column in data, for example 'x,y,y'. \
            Default to 'y' for one column and 'x,y' for two columns. \
            Plot attributes can be appended after 'y', like 'ythick;cr'. \
            By default, assume all columns are y. \
            ")
    parser.add_argument('-t', '--title',
                        help='title of the graph',
                        default='')
    parser.add_argument('--xlabel',
                        help='x label', type=six.text_type)
    parser.add_argument('--ylabel',
                        help='y label', type=six.text_type)
    parser.add_argument('--xlim',
                        help='x lim', type=float, nargs=2)
    parser.add_argument('--ylim',
                        help='y lim', type=float, nargs=2)
    parser.add_argument('-s', '--scale',
                        help='scale of each y, separated by comma')
    parser.add_argument('--annotate-maximum',
                        help='annonate maximum value in graph',
                        action='store_true')
    parser.add_argument('--annotate-minimum',
                        help='annonate minimum value in graph',
                        action='store_true')
    parser.add_argument('--xkcd',
                        help='xkcd style',
                        action='store_true')
    parser.add_argument('--decay',
                        help='exponential decay rate to smooth Y',
                        type=float, default=0)
    parser.add_argument('-l', '--legend',
                        help='legend for each y')
    parser.add_argument('-d', '--delimeter',
                        help='column delimeter', default='\t')

    global args
    args = parser.parse_args()

    if not args.show and not args.output:
        args.show = True


def filter_valid_range(points, rect):
    """rect = (min_x, max_x, min_y, max_y)"""
    ret = []
    for x, y in points:
        if x >= rect[0] and x <= rect[1] and y >= rect[2] and y <= rect[3]:
            ret.append((x, y))
    if len(ret) == 0:
        ret.append(points[0])
    return ret


def exponential_smooth(data, alpha):
    """ smooth data by alpha. returned a smoothed version"""
    ret = np.copy(data)
    now = data[0]
    for k in range(len(data)):
        ret[k] = now * alpha + data[k] * (1 - alpha)
        now = ret[k]
    return ret


def annotate_min_max(data_x, data_y, ax):
    max_x, min_x = max(data_x), min(data_x)
    max_y, min_y = max(data_y), min(data_y)
    x_range = max_x - min_x
    y_range = max_y - min_y
    x_max, y_max = data_y[0], data_y[0]
    x_min, y_min = data_x[0], data_y[0]

    for i in range(1, len(data_x)):
        if data_y[i] > y_max:
            y_max = data_y[i]
            x_max = data_x[i]
        if data_y[i] < y_min:
            y_min = data_y[i]
            x_min = data_x[i]

    rect = ax.axis()
    if args.annotate_maximum:
        text_x, text_y = filter_valid_range([
            (x_max + 0.05 * x_range,
                y_max + 0.025 * y_range),
            (x_max - 0.05 * x_range,
                y_max + 0.025 * y_range),
            (x_max + 0.05 * x_range,
                y_max - 0.025 * y_range),
            (x_max - 0.05 * x_range,
                y_max - 0.025 * y_range)],
            rect)[0]
        ax.annotate('maximum ({:d},{:.3f})' . format(int(x_max), y_max),
                    xy=(x_max, y_max),
                    xytext=(text_x, text_y),
                    arrowprops=dict(arrowstyle='->'))
    if args.annotate_minimum:
        text_x, text_y = filter_valid_range([
            (x_min + 0.05 * x_range,
                y_min - 0.025 * y_range),
            (x_min - 0.05 * x_range,
                y_min - 0.025 * y_range),
            (x_min + 0.05 * x_range,
                y_min + 0.025 * y_range),
            (x_min - 0.05 * x_range,
                y_min + 0.025 * y_range)],
            rect)[0]
        ax.annotate('minimum ({:d},{:.3f})' . format(int(x_min), y_min),
                    xy=(x_min, y_min),
                    xytext=(text_x, text_y),
                    arrowprops=dict(arrowstyle='->'))
        # ax.annotate('{:.3f}' . format(y_min),
        # xy = (x_min, y_min),
        # xytext = (text_x, text_y),
        # arrowprops = dict(arrowstyle = '->'))


def plot_args_from_column_desc(desc):
    if not desc:
        return {}
    ret = {}
    desc = desc.split(';')
    if 'thick' in desc:
        ret['lw'] = 5
    if 'dash' in desc:
        ret['ls'] = '--'
    for v in desc:
        if v.startswith('c'):
            ret['color'] = v[1:]
    return ret


def do_plot(data_xs, data_ys):
    """
    data_xs: list of 1d array, either of size 1 or size len(data_ys)
    data_ys: list of 1d array
    """
    fig = plt.figure(figsize=(16.18 / 1.2, 10 / 1.2))
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    nr_y = len(data_ys)
    y_column = args.y_column

    # parse legend and y-scale
    if args.legend:
        legends = args.legend.split(',')
        assert len(legends) == nr_y
    else:
        legends = None  # range(nr_y) #None
    if args.scale:
        scale = map(float, args.scale.split(','))
        assert len(scale) == nr_y
    else:
        scale = [1.0] * nr_y

    for yidx in range(nr_y):
        plotargs = plot_args_from_column_desc(y_column[yidx][1:])
        now_scale = scale[yidx]
        data_y = data_ys[yidx] * now_scale
        leg = legends[yidx] if legends else None
        if now_scale != 1:
            leg = "{}*{}".format(now_scale if int(now_scale) != now_scale else int(now_scale), leg)
        data_x = data_xs[0] if len(data_xs) == 1 else data_xs[yidx]
        assert len(data_x) >= len(data_y), \
            "x column is shorter than y column! {} < {}".format(
            len(data_x), len(data_y))
        truncate_data_x = data_x[:len(data_y)]
        p = plt.plot(truncate_data_x, data_y, label=leg, **plotargs)

        c = p[0].get_color()
        plt.fill_between(truncate_data_x, data_y, alpha=0.1, facecolor=c)

        # ax.set_aspect('equal', 'datalim')
        # ax.spines['right'].set_color('none')
        # ax.spines['left'].set_color('none')
        # plt.xticks([])
        # plt.yticks([])

        if args.annotate_maximum or args.annotate_minimum:
            annotate_min_max(truncate_data_x, data_y, ax)

    if args.xlabel:
        plt.xlabel(args.xlabel, fontsize='xx-large')
    if args.ylabel:
        plt.ylabel(args.ylabel, fontsize='xx-large')
    if args.xlim:
        plt.xlim(args.xlim[0], args.xlim[1])
    if args.ylim:
        plt.ylim(args.ylim[0], args.ylim[1])
    plt.legend(loc='best', fontsize='xx-large')

    # adjust maxx
    minx, maxx = min(data_x), max(data_x)
    new_maxx = maxx + (maxx - minx) * 0.05
    plt.xlim(minx, new_maxx)

    for label in chain.from_iterable(
            [ax.get_xticklabels(), ax.get_yticklabels()]):
        label.set_fontproperties(fontm.FontProperties(size=15))

    ax.grid(color='gray', linestyle='dashed')

    plt.title(args.title, fontdict={'fontsize': '20'})

    if args.output != '':
        plt.savefig(args.output, bbox_inches='tight')
    if args.show:
        plt.show()


def main():
    get_args()
    # parse input args
    if args.input == STDIN_FNAME:
        fin = sys.stdin
    else:
        fin = open(args.input)
    all_inputs = fin.readlines()
    if args.input != STDIN_FNAME:
        fin.close()

    # parse column format
    nr_column = len(all_inputs[0].rstrip('\n').split(args.delimeter))
    if args.column is None:
        column = ['y'] * nr_column
    else:
        column = args.column.strip().split(',')
    for k in column:
        assert k[0] in ['x', 'y']
    assert nr_column == len(column), "Column and data doesn't have same length. {}!={}".format(nr_column, len(column))
    args.y_column = [v for v in column if v[0] == 'y']
    args.y_column_idx = [idx for idx, v in enumerate(column) if v[0] == 'y']
    args.x_column = [v for v in column if v[0] == 'x']
    args.x_column_idx = [idx for idx, v in enumerate(column) if v[0] == 'x']
    nr_x_column = len(args.x_column)
    nr_y_column = len(args.y_column)
    if nr_x_column > 1:
        assert nr_x_column == nr_y_column, \
            "If multiple x columns are used, nr_x_column must equals to nr_y_column"

    # read and parse data
    data = [[] for _ in range(nr_column)]
    ended = defaultdict(bool)
    for lineno, line in enumerate(all_inputs):
        line = line.rstrip('\n').split(args.delimeter)
        assert len(line) <= nr_column, \
            """One row have too many columns (separated by {})!
Line: {}""".format(repr(args.delimeter), line)
        for idx, val in enumerate(line):
            if val == '':
                ended[idx] = True
                continue
            else:
                val = float(val)
                assert not ended[idx], "Column {} has hole!".format(idx)
                data[idx].append(val)

    data_ys = [data[k] for k in args.y_column_idx]
    length_ys = [len(t) for t in data_ys]
    print("Length of each column:", length_ys)
    max_ysize = max(length_ys)

    if nr_x_column:
        data_xs = [data[k] for k in args.x_column_idx]
    else:
        data_xs = [list(range(1, max_ysize + 1))]

    for idx, data_y in enumerate(data_ys):
        data_ys[idx] = np.asarray(data_y)
        if args.decay != 0:
            data_ys[idx] = exponential_smooth(data_y, args.decay)
        # if idx == 0:   # TODO allow different decay for each y
            # data_ys[idx] = exponential_smooth(data_y, 0.5)
    for idx, data_x in enumerate(data_xs):
        data_xs[idx] = np.asarray(data_x)

    if args.xkcd:
        with plt.xkcd():
            do_plot(data_xs, data_ys)
    else:
        do_plot(data_xs, data_ys)


if __name__ == '__main__':
    main()
