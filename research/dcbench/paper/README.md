# Paper: "Is the Hierarchy Doing Anything? Main-Network Collapse in Hierarchical Byte-Level Language Models"

* `main.pdf` — compiled paper (8 pages incl. appendix).
* `main.tex`, `refs.bib` — source (references checked against arXiv metadata).
* `make_figures.py` — regenerates every figure in `figures/` and `figures/numbers.json` from the
  per-run results in `../results/` (run from `research/dcbench`: `python paper/make_figures.py`).

Build:

```bash
cd research/dcbench && python paper/make_figures.py && cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The author block is anonymous (double-blind format); Appendix D discloses AI assistance — edit
both as appropriate for the venue.
