#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: test_examples.py
# Author: Patrick Wieschollek <mail@patwie.com>

import sys
import subprocess
import shlex
from threading import Timer
from termcolor import colored, cprint


class SurviveException(Exception):
    pass


def test_example(script_name, timeout_sec=20):
    """Run a python script for some seconds and test whether it survives the timeout or not.

    Args:
        script_name (string): relative path to script from root
        timeout_sec (int, optional): time which the scripts has to survive to pass the test

    Raises:
        SurviveException: contains possible stderr
    """
    cmd = "python %s" % script_name
    proc = subprocess.Popen(shlex.split(cmd), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    timer = Timer(timeout_sec, lambda p: p.kill(), [proc])
    try:
        timer.start()
        stdout, stderr = proc.communicate()
        if len(stderr) is not 0:
            stderr = "\n".join([" " * 10 + v for v in stderr.split("\n")])
            raise SurviveException(stderr)
    finally:
        timer.cancel()


EXAMPLES_TO_TEST = ["examples/mnist-convnet.py"]

total = len(EXAMPLES_TO_TEST)
passed = 0
failed = []
max_name_length = max([len(v) for v in EXAMPLES_TO_TEST]) + 10

for example_name in EXAMPLES_TO_TEST:
    string = "test %s %s" % (example_name, " " * (max_name_length - len(example_name)))
    sys.stdout.write(colored(string, 'yellow', attrs=['bold']))
    try:
        test_example(example_name)
        cprint("... works", 'green', attrs=['bold'])
        passed += 1
    except Exception as stderr_message:
        cprint("... failed", 'red', attrs=['bold'])
        print(stderr_message)
        failed.append(example_name)

print("\n\n")
cprint("Summary:    TEST passed %i / %i" % (passed, len(EXAMPLES_TO_TEST)), 'yellow', attrs=['bold'])
if total != passed:
    print("The following script failed:")
    for failed_script in failed:
        print("  - %s" % failed_script)
    sys.exit(1)
else:
    sys.exit(0)
