#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: visualqa.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ..base import DataFlow
from ...utils.timer import timed_operation
from six.moves import zip, map
from collections import Counter
import json

__all__ = ['VisualQA']


def read_json(fname):
    f = open(fname)
    ret = json.load(f)
    f.close()
    return ret


class VisualQA(DataFlow):
    """
    `Visual QA <http://visualqa.org/>`_ dataset.
    It simply reads q/a json file and produce q/a pairs in their original format.
    """

    def __init__(self, question_file, annotation_file):
        with timed_operation('Reading VQA JSON file'):
            qobj, aobj = list(map(read_json, [question_file, annotation_file]))
            self.task_type = qobj['task_type']
            self.questions = qobj['questions']
            self._size = len(self.questions)

            self.anno = aobj['annotations']
            assert len(self.anno) == len(self.questions), \
                "{}!={}".format(len(self.anno), len(self.questions))
            self._clean()

    def _clean(self):
        for a in self.anno:
            for aa in a['answers']:
                del aa['answer_id']

    def size(self):
        return self._size

    def get_data(self):
        for q, a in zip(self.questions, self.anno):
            assert q['question_id'] == a['question_id']
            yield [q, a]

    def get_common_answer(self, n):
        """ Get the n most common answers (could be phrases)
            n=3000 ~= thresh 4
        """
        cnt = Counter()
        for anno in self.anno:
            cnt[anno['multiple_choice_answer'].lower()] += 1
        return [k[0] for k in cnt.most_common(n)]

    def get_common_question_words(self, n):
        """ Get the n most common words in questions
            n=4600 ~= thresh 6
        """
        from nltk.tokenize import word_tokenize  # will need to download 'punckt'
        cnt = Counter()
        for q in self.questions:
            cnt.update(word_tokenize(q['question'].lower()))
        del cnt['?']    # probably don't need this
        ret = cnt.most_common(n)
        return [k[0] for k in ret]


if __name__ == '__main__':
    vqa = VisualQA('/home/wyx/data/VQA/MultipleChoice_mscoco_train2014_questions.json',
                   '/home/wyx/data/VQA/mscoco_train2014_annotations.json')
    for k in vqa.get_data():
        print(json.dumps(k))
        break
    vqa.get_common_answer(100)
