# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf

from ..utils import logger
from ..train.trainers import NoOpTrainer
from .param import ScheduledHyperParamSetter, ObjAttrParam


class ParamObject(object):
    """
    An object that holds the param to be set, for testing purposes.
    """
    PARAM_NAME = 'param'

    def __init__(self):
        self.param_history = {}
        self.__dict__[self.PARAM_NAME] = 1.0

    def __setattr__(self, name, value):
        if name == self.PARAM_NAME:
            self._set_param(value)
        super(ParamObject, self).__setattr__(name, value)

    def _set_param(self, value):
        self.param_history[self.trainer.global_step] = value


class ScheduledHyperParamSetterTest(unittest.TestCase):
    def setUp(self):
        self._param_obj = ParamObject()

    def tearDown(self):
        tf.reset_default_graph()

    def _create_trainer_with_scheduler(self, scheduler,
                                       steps_per_epoch, max_epoch, starting_epoch=1):
        trainer = NoOpTrainer()
        tf.get_variable(name='test_var', shape=[])
        self._param_obj.trainer = trainer
        trainer.train_with_defaults(
            callbacks=[scheduler],
            extra_callbacks=[],
            monitors=[],
            steps_per_epoch=steps_per_epoch,
            max_epoch=max_epoch,
            starting_epoch=starting_epoch
        )
        return self._param_obj.param_history

    def testInterpolation(self):
        scheduler = ScheduledHyperParamSetter(
            ObjAttrParam(self._param_obj, ParamObject.PARAM_NAME),
            [(30, 0.3), (40, 0.4), (50, 0.5)], interp='linear', step_based=True)
        history = self._create_trainer_with_scheduler(scheduler, 10, 50, starting_epoch=20)
        self.assertEqual(min(history.keys()), 30)
        self.assertEqual(history[30], 0.3)
        self.assertEqual(history[40], 0.4)
        self.assertEqual(history[45], 0.45)

    def testSchedule(self):
        scheduler = ScheduledHyperParamSetter(
            ObjAttrParam(self._param_obj, ParamObject.PARAM_NAME),
            [(10, 0.3), (20, 0.4), (30, 0.5)])
        history = self._create_trainer_with_scheduler(scheduler, 1, 50)
        self.assertEqual(min(history.keys()), 10)
        self.assertEqual(len(history), 3)

    def testStartAfterSchedule(self):
        scheduler = ScheduledHyperParamSetter(
            ObjAttrParam(self._param_obj, ParamObject.PARAM_NAME),
            [(10, 0.3), (20, 0.4), (30, 0.5)])
        history = self._create_trainer_with_scheduler(scheduler, 1, 92, starting_epoch=90)
        self.assertEqual(len(history), 0)

    def testWarningStartInTheMiddle(self):
        scheduler = ScheduledHyperParamSetter(
            ObjAttrParam(self._param_obj, ParamObject.PARAM_NAME),
            [(10, 0.3), (20, 0.4), (30, 0.5)])
        with self.assertLogs(logger=logger._logger, level='WARNING'):
            self._create_trainer_with_scheduler(scheduler, 1, 21, starting_epoch=20)

    def testNoWarningStartInTheMiddle(self):
        scheduler = ScheduledHyperParamSetter(
            ObjAttrParam(self._param_obj, ParamObject.PARAM_NAME),
            [(10, 0.3), (20, 1.0), (30, 1.5)])
        with unittest.mock.patch('tensorpack.utils.logger.warning') as warning:
            self._create_trainer_with_scheduler(scheduler, 1, 22, starting_epoch=21)
        self.assertFalse(warning.called)


if __name__ == '__main__':
    unittest.main()
