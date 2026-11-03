# AGENTS.md — `tests/integration/`

- These tests exercise the real upstream parser and the real PyMDP agent. Do not
  add a mock, a synthetic parser or a canned response.
- Never convert a missing runtime into a skip. The `gnn`/`full` profiles must
  refuse with `GNN_PROFILE_RUNTIME_REQUIRED`; a required profile with skips
  exits as a failure.
- Keep the module list in
  [`../../scripts/validation_profiles.py`](../../scripts/validation_profiles.py)
  (`GNN_MODULES`) in step with the files here; the CI shard plan requires every
  node in exactly one profile.
- Temporary storage must stay outside the checkout; `tests/conftest.py` enforces
  this through `TMPDIR`/`--basetemp`.
- Assertions must bind the observed bytes and digests, not just return codes.
  A successful subprocess with an unverified output is not evidence.
- Do not make these tests depend on network access beyond the caller-provisioned
  runtime.
- Parent rules: [../AGENTS.md](../AGENTS.md). GNN boundary:
  [`../../src/fedference/gnn/AGENTS.md`](../../src/fedference/gnn/AGENTS.md).
