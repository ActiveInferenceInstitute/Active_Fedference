# References {#sec:references}

The bibliography lives in
[project bibliography source](https://github.com/ActiveInferenceInstitute/Active_Fedference/blob/main/manuscript/references.bib)
and is read by Pandoc during rendering. The combined PDF path invokes
`--natbib`, so citation markers become LaTeX citation commands resolved against
the bib file. HTML, reveal.js, and other non-LaTeX reader surfaces use
`--citeproc` against that same file. This is one bibliography with
format-specific consumers, not two metadata sources. Titles retain the source's
original spelling.

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
