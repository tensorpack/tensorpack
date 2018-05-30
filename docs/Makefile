# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = tensorpack
SOURCEDIR     = .
BUILDDIR      = build

.PHONY: help Makefile docset clean

all: html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docset: html
	convert _static/favicon.ico ./favicon.png
	doc2dash -d ./ -n $(SPHINXPROJ) --enable-js --force $(BUILDDIR)/html/ -I tutorial/index.html -i favicon.png
	tar czvf tensorpack.docset.tgz tensorpack.docset

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf build
