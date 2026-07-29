# Minimal Makefile for Sphinx documentation
SPHINXOPTS      ?=
SPHINXBUILD     ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR       = .
BUILDDIR        = _build
PORT            ?= 8003

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional targets:"
	@echo "  serve       build and serve the docs with auto-build and live reload"

# Serve documentation with auto-build and live reload
serve:
	@echo "Starting auto-build server at http://0.0.0.0:$(PORT)"
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host 0.0.0.0 \
		--port $(PORT) \
		--watch $(SOURCEDIR) \
		--re-ignore ".*\.(pyc|pyo|pyd|git)"

clean:
	rm -rf $(BUILDDIR)

.PHONY: help Makefile serve clean

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
