# References {#sec:references}

The bibliography lives in [`manuscript/references.bib`](references.bib) and is
read by Pandoc during the PDF render. The build pipeline invokes Pandoc with
`--natbib`, so every Pandoc citation marker in the manuscript is rewritten to
the appropriate LaTeX citation command and resolved against the bib file. Titles
in the bib file are reproduced verbatim, including any British spellings,
because they are quotations of the original sources.

The standalone checkout provides a local cross-reference gate for citation
labels and all manuscript references:

```bash
uv run --locked pytest tests/test_xref_integrity.py -q
```

To validate that `references.bib` is syntactically clean and contains the
required fields per entry type, the stricter citation validator is available
when the project is checked out under the template monorepo's
`projects/working/` (it is not on the standalone repo's own dependency graph),
invoked from the monorepo root with a monorepo-relative path:

```bash
uv run python -m infrastructure.reference.citation.cli validate \
    projects/working/active_fedference/manuscript/references.bib --strict
```
