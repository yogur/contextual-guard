# Contextual Guard

Start with [README](README.md). [docs/USAGE.md](docs/USAGE.md) owns the guard
contract and provider setup. [benchmark/README.md](benchmark/README.md) is the
entry point for datasets, results, methodology and scripts.

`src/jev_guard` and `src/jev_contracts` form the public wheel. `benchmark/src`
depends on the guard; keep that dependency direction. Packaged prompts are
authoritative. The host owns enforcement. Never execute tools from logged data.

Keep model behavior, prompts, labels and saved verdicts unchanged during release
cleanup. Do not read `.env` or run inference as a verification step. Full benchmark
runs are owner-run. New guard results retain token usage without cost estimates;
historical annotation accounting stays intact.

Checks: `uv sync --locked`, `uv run pytest -q`,
`uv run ruff check src benchmark/src tests benchmark/scripts examples`, and
`uv run ruff format --check src benchmark/src tests benchmark/scripts examples`.
The full offline check is documented in RELEASE. Full datasets are GitHub Release
assets because of their size; `benchmark/datasets/v1/` holds local copies and tracked
checksums/manifests. `benchmark/config/selection.toml` owns corpus selection. Python
build output belongs in `dist/`. Do not commit or publish without instruction.
