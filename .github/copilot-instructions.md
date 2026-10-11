# Copilot instructions

## Build and run

- This is a Python 3.11+ project managed with `uv`; `pyproject.toml` is the source of project metadata and dependencies.
- Install/sync the project environment with `uv sync`.
- Run the configured console script with `uv run building-reasoning-models-from-scratch`.
- Build the package with `uv build`.
- No test suite, test runner, or lint configuration is currently present. If tests are added using the standard-library `unittest`, run one test with `uv run python -m unittest tests.test_module.TestClass.test_method`; replace the dotted path with the test module and test name.

## Architecture

- This repository is an in-progress, from-scratch Qwen3 implementation, not a finished training or inference application. The package entry point in `src/building_reasoning_models_from_scratch/__init__.py` is currently a minimal console-script placeholder.
- `config/qwen_cfg.yaml` holds the Qwen3 0.6B model dimensions and related architecture settings. `utils/config_loader.py` loads YAML into attribute-accessible configuration objects.
- `networks/` contains model building blocks: grouped-query attention, rotary positional embeddings, RMS normalization, and KV-cache support. Attention is intended to compose these pieces and return an updated cache for incremental decoding.
- `models/qwen3.py` is the intended model orchestration layer; it is currently only a placeholder. `utils/qwen3_model_weights.py` is reserved for model-weight handling, and `tokenizers/qwen3tokenizer.py` is an unfinished tokenizer module.

## Repository-specific conventions

- Keep Qwen3 architecture parameters centralized in `config/qwen_cfg.yaml` rather than duplicating model dimensions in implementation code.
- Preserve the separation between reusable tensor/attention components in `networks/`, model-level composition in `models/`, and configuration/weight-loading utilities in `utils/`.
- Grouped-query attention distinguishes the query-head count from the key/value-group count; when changing attention behavior, preserve their divisibility relationship and account for the optional per-head dimension and Q/K normalization.
- Keep rotary embedding parameter generation and application in `networks/encodings.py`, and normalization behavior in `networks/norms.py`, rather than embedding those implementations in the model wrapper.
