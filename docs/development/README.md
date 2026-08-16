# Development documentation

How to modify code and tests in Active Fedference.

| Document | Purpose |
| --- | --- |
| [agent_instructions.md](agent_instructions.md) | Repository editing rules, visual QA, and pre-submit checklist |
| [quickstart.md](quickstart.md) | Runnable-in-order recipe: green baseline to rendered PDF |
| [testing_philosophy.md](testing_philosophy.md) | Zero-mock policy; coverage gate; schema/contract gates; proof of detection |
| [style_guide.md](style_guide.md) | Thin orchestrator, layer contract, show-not-tell |
| [modularity.md](modularity.md) | Cross-layer software, CLI, script, report, figure, and documentation extension contract |

These pages are themselves contract surfaces: `tests/test_docs_contract.py`
discovers every Markdown page under `docs/`, verifies that relative links
resolve, that documented `scripts/*.py` commands name real files, and that
stale claim language does not reappear.
