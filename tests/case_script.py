from abc import abstractproperty
import unittest
import subprocess
import shlex
import sys
import threading
import os
import shutil


class PythonScript(threading.Thread):
    """A wrapper to start a python script with timeout.

    To test the actual models even without GPUs we simply start them and
    test whether they survive a certain amount of time "timeout". This allows to
    test if all imports are correct and the computation graph can be built without
    run the entire model on the CPU.

    Attributes:
        cmd (str): command to execute the example with all flags (including python)
        p: process handle
        timeout (int): timeout in seconds
    """
    def __init__(self, cmd, timeout=10):
        """Prepare a python script

        Args:
            cmd (TYPE): command to execute the example with all flags (including python)
            timeout (int, optional): time in seconds the script has to survive
        """
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout

    def run(self):
        self.p = subprocess.Popen(shlex.split(self.cmd), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        self.out, self.err = self.p.communicate()

    def execute(self):
        """Execute python script in other process.

        Raises:
            SurviveException: contains the error message of the script if it terminated before timeout
        """
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            self.p.terminate()
            self.p.kill()  # kill -9
            self.join()
        else:
            # something unexpected happend here, this script was supposed to survive at leat the timeout
            if len(self.err) is not 0:
                stderr = "\n\n\n\n\n %s" % self.err
                raise AssertionError(stderr)


class TestPythonScript(unittest.TestCase):

    @abstractproperty
    def script(self):
        pass

    @staticmethod
    def clear_trainlog(script):
        script = os.path.basename(script)
        script = script[:-3]
        if os.path.isdir(os.path.join("train_log", script)):
            shutil.rmtree(os.path.join("train_log", script))

    def assertSurvive(self, script, args=None, timeout=10):  # noqa
        cmd = "python{} {}".format(sys.version_info.major, script)
        if args:
            cmd += " " + " ".join(args)
        PythonScript(cmd, timeout=timeout).execute()

    def setUp(self):
        TestPythonScript.clear_trainlog(self.script)

    def tearDown(self):
        TestPythonScript.clear_trainlog(self.script)
