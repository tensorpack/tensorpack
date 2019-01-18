
## Build the docs:

### Dependencies:
1. Python 3
2. Remove "tensorflow" from `requirements.txt` since you probably prefer to install TensorFlow by yourself.
3. `pip install -r requirements.txt`. Note that these requirements are different from tensorpack dependencies.

### Build HTML docs:
`make html`
will build the docs in `build/html`.

### Build Dash/Zeal docset

1. `pip install doc2dash`
2. `make docset` produces `tensorpack.docset`.

### Subscribe to docset updates in Dash/Zeal:

Add this feed in Dash/Zeal: `https://github.com/tensorpack/tensorpack/raw/master/docs/tensorpack.xml`.
