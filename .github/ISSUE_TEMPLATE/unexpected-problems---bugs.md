---
name: Unexpected Problems / Bugs
about: Report unexpected problems about Tensorpack or its examples.

---

If you're asking about an unexpected problem you met, use this template.
__PLEASE DO NOT DELETE THIS TEMPLATE, FILL IT__:

### 1. What you did:

(1) **If you're using examples, what's the command you run:**

(2) **If you're using examples, have you made any changes to the examples? Paste them here:**

(3) **If not using examples, tell us what you did here:**

Note that we may not be able to investigate it if there is no reproducible code.
It's always better to paste what you did instead of describing them.

### 2. What you observed:

(1) **Include the ENTIRE logs here:**

It's always better to paste what you observed instead of describing them.

A part of logs is sometimes enough, but it's always better to paste as much as possible.

You can run a command with `CMD 2>&1 | tee logs.txt` to save all stdout & stderr logs to one file.

(2) **Other observations, if any:**
For example, CPU/GPU utilization, output images, tensorboard curves, if relevant to your issue.

### 3. What you expected, if not obvious.
### 4. Your environment:
  + Python version:
  + TF version: `python -c 'import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)'`.
  + Tensorpack version: `python -c 'import tensorpack; print(tensorpack.__version__)'`.
      You can install Tensorpack master by `pip install -U git+https://github.com/ppwwyyxx/tensorpack.git`
      and see if your issue is already solved.
  + If you're not using tensorpack under a normal command line shell (e.g.,
    using an IDE or jupyter notebook), please retry under a normal command line shell. 
  + Hardware information, e.g. number of GPUs used.

About efficiency issues, PLEASE first read http://tensorpack.readthedocs.io/en/latest/tutorial/performance-tuning.html
