---
name: Unexpected Problems / Bugs
about: Report unexpected problems about Tensorpack or its examples.
title: Please read & provide the following information
labels: ''
assignees: ''

---

If you're asking about an unexpected problem which you do not know the root cause,
use this template. __PLEASE DO NOT DELETE THIS TEMPLATE, FILL IT__:

If you already know the root cause to your problem,
feel free to delete everything in this template.

### 1. What you did:

(1) **If you're using examples, what's the command you run:**

(2) **If you're using examples, have you made any changes to the examples? Paste `git status; git diff` here:**

(3) **If not using examples, help us reproduce your issue:**

  It's always better to copy-paste what you did than to describe them.

  Please try to provide enough information to let others __reproduce__ your issues.
  Without reproducing the issue, we may not be able to investigate it.

### 2. What you observed:

(1) **Include the ENTIRE logs here:**
```
<paste logs here>
```

It's always better to copy-paste what you observed instead of describing them.

It's always better to paste **as much as possible**, although sometimes a partial log is OK.

Tensorpack typically saves stdout to its training log.
If stderr is relevant, you can run a command with `my_command 2>&1 | tee logs.txt`
to save both stdout and stderr to one file.

(2) **Other observations, if any:**
For example, CPU/GPU utilization, output images, tensorboard curves, if relevant to your issue.

### 3. What you expected, if not obvious.

If you expect higher speed, please read
http://tensorpack.readthedocs.io/tutorial/performance-tuning.html
before posting.

If you expect the model to converge / work better, note that we do not help you on how to improve a model.
Only in one of the two conditions can we help with it:
(1) You're unable to reproduce the results documented in tensorpack examples.
(2) It indicates a tensorpack bug.

### 4. Your environment:

Paste the output of this command: `python -m tensorpack.tfutils.collect_env`
If this command failed, also tell us your version of Python/TF/tensorpack.

Note that:

  + You can install tensorpack master by `pip install -U git+https://github.com/tensorpack/tensorpack.git`
    and see if your issue is already solved.
  + If you're not using tensorpack under a normal command line shell (e.g.,
    using an IDE or jupyter notebook), please retry under a normal command line shell.

You may often want to provide extra information related to your issue, but
at the minimum please try to provide the above information __accurately__ to save effort in the investigation.
