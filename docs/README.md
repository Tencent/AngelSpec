# AngelSpec Documentation

Source for the AngelSpec documentation site, built with [Sphinx](https://www.sphinx-doc.org/)
and the [sphinx-book-theme](https://sphinx-book-theme.readthedocs.io/). Pages are written in
Markdown ([MyST](https://myst-parser.readthedocs.io/)).

## Build locally

```bash
cd docs
pip install -r requirements.txt

# One-off build → _build/html/index.html
make html

# Live-reload server (rebuilds on save); open the printed port
bash serve.sh          # or: make serve   (PORT=8080 make serve to change port)
```

## Writing guidelines

- **Markdown-first.** Most AngelSpec workflows are multi-node and RDMA-based, so examples are
  documented as Markdown that mirrors the scripts under `examples/`, rather than runnable
  notebooks.
- **Relative links only.** Link to other pages with relative paths (e.g. `../concepts/dflash.md`),
  never absolute site URLs, so links keep working across versions.
- **Keep code blocks in sync** with the actual `examples/*/run.sh` scripts and `configs/*.yaml`.
- **New page?** Add it to the matching `toctree` caption in `index.rst`.
- **Diagrams** can use [Mermaid](https://mermaid.js.org/) via the ```` ```{mermaid} ```` directive.

## Layout

| Section | Folder | Contents |
|---------|--------|----------|
| Get Started | `get_started/` | Motivation, installation, quickstart |
| Concepts | `concepts/` | Speculative decoding, disaggregated architecture, the draft-model family |
| Basic Usage | `basic_usage/` | Data prep, training, inference backends, checkpoint conversion |
| Advanced Features | `advanced_features/` | Multi-node, long-sequence USP, packing, OPD, customization |
| Operations | `operations/` | Ray cluster, debugging, performance metrics, logging |
| Examples | `examples/` | End-to-end walkthroughs, simplest to most advanced |
| Reference | `reference/` | Code architecture, full configuration reference |
