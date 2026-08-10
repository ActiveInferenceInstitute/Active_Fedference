# Publication accessibility contract

Active Fedference treats the validated HTML manuscript as its canonical
accessibility-enhanced reading surface. The combined manuscript PDF and slide
PDFs are convenience and archival surfaces. They remain release-checked for
structure, extracted text, references, layout warnings, and visual integrity,
but the current tracked combined PDF reports `Tagged: no`; the PDF surfaces
must not be described as tagged, accessible PDF, or PDF/UA-conformant.

This is a scoped engineering contract, not a declaration of WCAG conformance.
The automated rules cover deterministic properties that this repository can
enforce. A public accessibility claim still requires manual keyboard,
screen-reader, zoom/reflow, contrast, reading-order, and mathematical-content
review.

## HTML rules enforced in source

`src/publication/web_package.py::validate_web_package` fails every generated
HTML page unless it has:

- exactly one non-empty document language and title;
- exactly one focusable `<main id="main-content" tabindex="-1">` landmark;
- exactly one `.skip-link` targeting that landmark as the first interactive
  element;
- a non-empty `alt` attribute on every image;
- a `<figcaption>` for every figure containing an image;
- an `aria-label` on every full-size figure link;
- no duplicate element identifiers;
- no missing local assets, broken fragments, raw typed references, or leaked
  figure Markdown.

These rules support the semantic structure, text alternatives, page language,
page title, and bypass-block concerns in
[WCAG 2.2](https://www.w3.org/TR/WCAG22/). The skip-link pattern follows the
[W3C G1 technique](https://www.w3.org/WAI/WCAG22/Techniques/general/G1).
Passing the checks does not establish WCAG A, AA, or AAA conformance because
the validator does not determine alternative-text quality, contrast,
meaningful sequence, keyboard usability, assistive-technology behavior, or
whether visual encodings remain understandable without color.

Figure descriptions originate in manuscript captions and the figure registry,
not in generated HTML edits. Captions must identify the estimand, encodings,
units, uncertainty, source relation, and claim boundary without relying on
color alone. At release review, inspect whether the image alternative and
adjacent caption cause confusing repetition in at least one screen reader; if
so, repair the sibling renderer to produce a concise alternative with a
separate long description.

## PDF and slide boundary

`qpdf --check`, `pdftotext`, clean LaTeX logs, and raster inspection answer
important but different questions. They do not prove reading order, semantic
tagging, alternative descriptions, table structure, language metadata, or
assistive-technology interoperability. Tagged PDF supplies the semantic
structure used for those purposes; see the
[PDF Association accessibility resources](https://pdfa.org/accessibility/).

A future tagged-PDF promotion requires all of the following from the sibling
rendering producer:

1. tagged manuscript and slide files produced from source, never post-hoc
   hand-edited reviewer artifacts;
2. document language, title, heading hierarchy, lists, tables, equations,
   figures, alternatives, artifacts, and logical reading order represented in
   the tag tree;
3. `pdfinfo` reporting `Tagged: yes`;
4. a dedicated PDF/UA conformance validator with its full report retained;
5. keyboard, reflow/zoom, text extraction, and at least one screen-reader pass;
6. source-bound render and release receipts regenerated after the producer
   change.

Until every item passes, publication prose must say “HTML
accessibility-enhanced; PDFs structurally and visually validated but untagged,”
not “fully accessible,” “WCAG conformant,” or “PDF/UA conformant.”

## Release review sequence

Run source validation first, then generate analysis, hydrate the manuscript,
render through the sibling producer, prepare the web package, and execute:

```bash
uv run --locked python scripts/validate_web_package.py
uv run --locked python scripts/validate_rendered_surfaces.py
pdfinfo output/pdf/active_fedference_combined.pdf | grep '^Tagged:'
```

Finally perform the manual checks above in a real browser and PDF reader.
Record findings with the render receipt. Do not repair generated files in
`output/`; fix the manuscript, figure producer, or sibling renderer and rerun
the producer chain.
