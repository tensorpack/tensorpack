
## Build the docs:

### Dependencies:
1. Python3
2. `pip install -r requirements.txt`. These requirements are different from tensorpack dependencies.

### Build HTML docs:
`make html`
will build the docs in `build/html`.

### Build Dash/Zeal docset

1. `pip install doc2dash`
2. `make docset` produces `tensorpack.docset`.
