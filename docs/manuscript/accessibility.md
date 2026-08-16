# Publication accessibility contract

Active Fedference treats the validated HTML manuscript as its canonical,
accessibility-enhanced reading surface. The combined manuscript PDF is also
generated through the source-controlled LuaLaTeX/tagpdf path requested by
`metadata.tagged_pdf: true`. Its release gate requires `pdfinfo` to report
`Tagged: yes`, qpdf JSON to expose a non-empty catalog `/Lang` and a
`StructTreeRoot`, and the source-bound language check to pass. Some Poppler
builds omit the language line from `pdfinfo`; that is why the validator checks
the PDF catalog as well. Figure alternatives are bound from
`src/figures/_metadata.py` through the figure registry. Slide PDFs are separate
Beamer outputs and are not automatically promoted to the tagged-PDF contract.

Tagged structure is not the same as PDF/UA certification. A PDF/UA claim is
allowed only when the retained veraPDF report and manual reading-order,
keyboard, reflow/zoom, mathematical-content, and screen-reader checks pass.
If those checks are not complete, the accurate status is “tagged PDF producer
enabled; PDF/UA conformance not established.” The public v1.0.2 artifact may
retain older surface properties; the current source and its regenerated
reviewer snapshot are the evidence for this contract.

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

`qpdf --check`, `pdftotext`, clean LaTeX logs, raster inspection, and
`pdfinfo Tagged: yes` answer important but different questions. They do not
prove PDF/UA reading order, table structure, or assistive-technology
interoperability. Tagged PDF supplies a semantic structure and figure
alternatives for those purposes; see the
[PDF Association accessibility resources](https://pdfa.org/accessibility/).

A source-current tagged manuscript PDF requires all of the following from the
sibling rendering producer:

1. the combined manuscript is produced from source, never post-hoc hand-edited;
2. the source document language and title are present;
3. figure alternatives are complete in the typed registry and bound to every
   embedded figure;
4. `pdfinfo` reports `Tagged: yes`, qpdf exposes `/Lang` and a
   `StructTreeRoot`, and the source-bound language check passes;
5. the source-bound render and release receipts are regenerated after any
   producer change.

Until the additional veraPDF and manual checks pass, publication prose must
say “HTML accessibility-enhanced; tagged PDF structure verified, PDF/UA
conformance not established,” not “fully accessible,” “WCAG conformant,” or
“PDF/UA conformant.”

## Release review sequence

Run source validation first, then generate analysis, hydrate the manuscript,
render through the sibling producer, prepare the web package, and execute:

```bash
uv run --locked python scripts/validate_web_package.py
uv run --locked python scripts/validate_rendered_surfaces.py
pdfinfo output/pdf/active_fedference_combined.pdf | grep '^Tagged:'
qpdf --json --json-stream-data=none output/pdf/active_fedference_combined.pdf \
  | grep -Eq '"/(Lang|StructTreeRoot)"'
verapdf --format text --flavour ua2 \
  output/pdf/active_fedference_combined.pdf > .tmp/verapdf-ua2.txt
```

Treat the veraPDF command as a conformance probe, not as a pass by invocation;
inspect and retain its complete report. Finally perform the manual checks above
in a real browser and PDF reader. Record findings with the render receipt. Do
not repair generated files in `output/`; fix the manuscript, figure producer,
or sibling renderer and rerun the producer chain.
