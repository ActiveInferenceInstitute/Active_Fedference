# AGENTS.md — `examples/`

Read [README.md](README.md) first; it is the numbered ladder and the expected
outputs. The application path is
[`../docs/application-guide.md`](../docs/application-guide.md).

- Keep every program runnable with the default locked environment:
  `uv sync --locked` then `uv run --locked python examples/<name>.py`. Do not
  require the `bnn` extra or Torch.
- Every program writes one deterministic JSON document to standard output. The
  three programs that write files require an explicit, new or empty
  `--output-dir` and never clean a caller's directory.
- Use the canonical input
  [`data/labeled_aggregation_request.json`](data/labeled_aggregation_request.json);
  do not add a duplicate request file.
- Keep the socket example loopback-only and its HMAC key an explicit non-secret
  fixture. Never add network access or a real credential.
- [`../tests/test_examples.py`](../tests/test_examples.py) executes every program
  as a real child process from outside the checkout and pins stable JSON
  semantics, direct/process/socket parity, replay tamper controls and receipt
  verification. A new example must be added to that test and to the README
  ladder.
- Examples must stay thin: call the public package or CLI boundary instead of
  reimplementing aggregation math.
- Parent rules: [../AGENTS.md](../AGENTS.md). Extension rules:
  [`../docs/development/modularity.md`](../docs/development/modularity.md).
