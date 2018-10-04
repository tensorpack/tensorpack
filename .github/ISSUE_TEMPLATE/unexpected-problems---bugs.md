---
name: Unexpected Problems / Bugs
about: Report unexpected problems about Tensorpack or its examples.

---

__PLEASE ALWAYS INCLUDE__:
1. What you did:
  + If you're using examples:
    + What's the command you run:
    + Have you made any changes to the examples? Paste them if any:
  + If not, tell us what you did that may be relevant.
    But we may not investigate it if there is no reproducible code.
  + Better to paste what you did instead of describing them.
2. What you observed, including but not limited to the __entire__ logs.
  + Better to paste what you observed instead of describing them.
3. What you expected, if not obvious.
4. Your environment:
  + Python version.
  + TF version: `python -c 'import tensorflow as tf; print(tf.GIT_VERSION, tf.VERSION)'`.
  + Tensorpack version: `python -c 'import tensorpack; print(tensorpack.__version__)'`.
      You can install Tensorpack master by `pip install -U git+https://github.com/ppwwyyxx/tensorpack.git`.:
  + Hardware information, e.g. number of GPUs used.

About efficiency issues, PLEASE first read http://tensorpack.readthedocs.io/en/latest/tutorial/performance-tuning.html
