# Minimal makefile for Sphinx documentation.
# Usage: `make -C docs html` (output in docs/_build/html/index.html).

SPHINXBUILD ?= sphinx-build
SOURCEDIR   = .
BUILDDIR    = _build

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
