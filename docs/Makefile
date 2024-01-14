# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "source" "build" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	cd "source"; $(SPHINXBUILD) -M $@ "." "../build" $(SPHINXOPTS) $(O)

html-noplot:
	cd "source"; $(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) . ../build/html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

clean:
	rm -rf build/*
	rm -rf source/auto_examples/*
	rm -rf source/gen_modules/*
	rm -rf source/sg_execution_times.rst
