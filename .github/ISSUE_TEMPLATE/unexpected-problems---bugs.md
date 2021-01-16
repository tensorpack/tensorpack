---
name: Unexpected Problems / Bugs
about: Report unexpected problems about Tensorpack or its examples.
title: Please read & provide the following information
labels: ''
assignees: ''
issue_body: true
inputs:
  - type: description
    attributes:
      value: Please fill out this form as completely as possible, unless you already find out the root cause to your issue.
  - type: textarea
    attributes:
      label: Your observation
      description: Provide __full__ logs of the observation, incomplete logs are less helpful. Please also provide other relevant observations if not in logs. If needed, images can be provided in the issue body below
      required: true
  - type: textarea
    attributes:
      label: Expected Behavior
      description: If there are no obvious failures, provide a clear and concise description of what you expected to happen. 
      required: false
  - type: textarea
    attributes:
      label: Your Environment
      description: provide the output of `python -m tensorpack.tfutils.collect_env`. If this fails, provide the logs and tell us how you install TensorFlow/tensorpack.
      required: true

---


### How to reproduce the issue:

(1) **If you're using examples, what's the command you run:**

(2) **If you're using examples, have you made any changes to the examples? Paste `git status; git diff` here:**

(3) **If not using examples, help us reproduce your issue:**

  It's always better to copy-paste what you did than to describe them.

  Please try to provide enough information to let others __reproduce__ your issues.
  Without reproducing the issue, we may not be able to investigate it.

### Other Information:
You may often want to provide extra information related to your issue, but
at the minimum please try to provide the above information __accurately__ to save effort in the investigation.
