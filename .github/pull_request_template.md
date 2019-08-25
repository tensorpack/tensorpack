Thanks for your contribution!

Unless you want to send a simple several lines of PR that can be easily merged, please note the following:

* If you want to add a new feature,
  please open an issue first and indicate that you want to contribute.

  There are features that we prefer to not add to tensorpack, e.g. symbolic models
  (see details at https://tensorpack.readthedocs.io/tutorial/symbolic.html).
  Therefore it's good to have a discussion first.

* If you want to add a new example, please note that:

  1. We prefer to not have an example that is too similar to existing ones in terms of the tasks.

  2. Examples have to be able to reproduce (preferrably in some measurable metrics) published or well-known experiments and results.

* Please run `flake8 .` under the root of this repo to lint your code, and make sure the command produces no output.
