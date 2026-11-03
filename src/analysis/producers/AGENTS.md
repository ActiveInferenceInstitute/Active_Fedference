# AGENTS.md — `src/analysis/producers/`

- Producers return payloads. Never write files, create directories, append to a
  registry or perform schema validation here.
- Import producer functions directly from `workflow.py`; do not add a dynamic
  discovery layer. The ownership test expects direct family imports.
- Take every seed, size and knob from the passed `ExperimentConfig`. No
  module-level defaults and no global `np.random`.
- Keep family boundaries: a new report lands in the matching producer or a new
  family module, not in `workflow.py`.
- Add the matching schema module and a test when the payload shape changes.
- Parent rules: [../../AGENTS.md](../../AGENTS.md). The analysis-level contract
  is [../AGENTS.md](../AGENTS.md).
