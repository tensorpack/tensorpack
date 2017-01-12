#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: test_examples.py
# Author: Patrick Wieschollek <mail@patwie.com>

import sys
import subprocess
import shlex
import threading
from termcolor import colored, cprint


COMMANDS_TO_TEST = ["python examples/mnist-convnet.py"]


class SurviveException(Exception):
    """Exception when process is already terminated
    """
    pass


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
                stderr = "\n".join([" " * 10 + v for v in self.err.split("\n")])
                raise SurviveException(stderr)


examples_total = len(COMMANDS_TO_TEST)
examples_passed = 0
examples_failed = []
max_name_length = max([len(v) for v in COMMANDS_TO_TEST]) + 10

cprint("Test %i python scripts with timeout" % (examples_total), 'yellow', attrs=['bold'])

for example_name in COMMANDS_TO_TEST:
    string = "test: %s %s" % (example_name, " " * (max_name_length - len(example_name)))
    sys.stdout.write(colored(string, 'yellow', attrs=['bold']))
    try:
        PythonScript(example_name).execute()
        cprint("... works", 'green', attrs=['bold'])
        examples_passed += 1
    except Exception as stderr_message:
        cprint("... examples_failed", 'red', attrs=['bold'])
        print(stderr_message)
        examples_failed.append(example_name)

print("\n\n")
cprint("Summary:    TEST examples_passed %i / %i" % (examples_passed, examples_total), 'yellow', attrs=['bold'])
if examples_total != examples_passed:
    print("The following script examples_failed:")
    for failed_script in examples_failed:
        print("  - %s" % failed_script)
    sys.exit(1)
else:
    sys.exit(0)
