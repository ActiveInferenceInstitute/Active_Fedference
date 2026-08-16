# `fedference_cli` — installed orchestration boundary

`fedference_cli` is the installed `fedference` command-line boundary for
source-bound research runs. It is intentionally small at the package surface:
the public `main()` entry point remains stable while the implementation is
split by responsibility. The package does not own research mathematics, report
schemas, or publication rendering.

## Module map

| Module | Responsibility | Must not become responsible for |
| --- | --- | --- |
| `__init__.py` | Stable compatibility facade exporting `main` and the historical `_report_fallbacks` helper | Argument parsing, experiment dispatch, filesystem writes, or domain imports beyond the facade imports |
| `_parser.py` | `argparse` grammar, command selection, process-facing error mapping, and the timestamp supplied to a run | Experiment algorithms, receipt construction, or direct file writes |
| `_commands.py` | Registry-backed dispatch for `run`/`benchmark`/`verify`/`replay` | Parser definition, duplicated aggregation math, or publication-snapshot writes |
| `_support.py` | Atomic JSON writes, project-root and output isolation, seed/control validation, registry summaries, and receipt construction | Selecting a study, choosing scientific parameters, or parsing command-line arguments |
| `__main__.py` | `python -m fedference_cli` process shim | Alternative command semantics |

The direction is deliberately one-way:

```text
fedference_cli.__main__
        -> fedference_cli.__init__
        -> fedference_cli._parser
             -> fedference_cli._commands
                  -> fedference domain/evidence/registry boundaries
             -> fedference_cli._support
```

The installed command is therefore useful for operators and CI while the
underlying functions remain importable for tests and library callers. Scripts
may invoke the CLI as a subprocess, but scripts must not reimplement these
checks or import private CLI helpers to perform research work.

## Stable public surface

```python
from fedference_cli import main

status = main(["list", "--json"])
```

The legacy import below remains available for compatibility with existing
tests and review tooling:

```python
from fedference_cli import _report_fallbacks
```

New integrations should prefer the typed APIs in `fedference` and its evidence
and registry modules. The CLI is an adapter for explicit process boundaries,
not the package's domain API.

## Adding a CLI workflow

1. Add or update the registry declaration in
   [`fedference/research_registry.py`](../fedference/research_registry.py),
   including the estimand, independent unit, falsifier, budget, MCSE target,
   comparison family, profile, and no-claim boundary.
2. Implement reusable execution in a domain or experiment module under
   [`fedference/`](../fedference/); keep it deterministic and independently
   testable.
3. Add only the dispatch branch and configuration serialization needed in
   [`_commands.py`](_commands.py). Keep validation and receipt mechanics in
   [`_support.py`](_support.py).
4. Add parser arguments in [`_parser.py`](_parser.py) only when the workflow
   needs a new caller-controlled value. Validate before creating the output
   directory.
5. Extend the zero-mock CLI tests in
   [`tests/test_fedference_cli.py`](../../tests/test_fedference_cli.py) and the
   relevant domain tests. A smoke run is implementation evidence, not
   confirmatory scientific evidence.
6. Update the user-facing command table in
   [`scripts/README.md`](../../scripts/README.md) only if a script changes;
   update the research and verification docs when the contract changes.

Every completed run must pass the typed report validator and write a
configuration-bound receipt. Run output belongs in an explicit, empty,
caller-owned directory outside the committed `output/` reviewer snapshot.

## Optional dependencies and safety

The default import path remains NumPy/SciPy-only. Torch/BNN support is selected
by the `bnn` extra and reports any device fallback explicitly. A CLI command
must not make optional dependencies part of the default import graph, silently
replace a failed algorithmic path, or claim CUDA, multi-host federation, or
confirmatory evidence from a CPU smoke run.

See the project-wide [modularity guide](../../docs/development/modularity.md),
the [API stability policy](../../docs/reference/api-stability.md), and the
[CLI contract tests](../../tests/test_fedference_cli.py).
