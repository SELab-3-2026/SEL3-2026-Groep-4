---
name: Run Tests
description: Instructions for executing the project test suite to verify code correctness.
---

# Skill: Run Tests

The goal of this skill is to verify that the project is functioning correctly after development changes. 

## Instructions

1. **Verify Environment**: The project operates dynamically with hardware acceleration (GPU via JAX) and uses `uv` for dependency/execution management.
2. **Execute Tests**:
   - To run basic verification (e.g., checking JAX initialization and hardware detection), run the test script:
     ```bash
     uv run python -m tests.test_jax_init
     ```
   - If a broader suite of modular tests is added (e.g., `pytest`), execute tests in the `tests/` directory with:
     ```bash
     uv run pytest tests/
     ```

## Important Considerations
- If you are running tests inside the local environment without a Devcontainer, verify whether `uv sync --frozen` (for CPU) or `uv sync --frozen --extra cuda` (for GPU) has been executed to avoid import errors.
- Do not run bare `python ...` without `uv run` locally, unless you are strictly operating inside a pre-activated `.venv` inside a Devcontainer.
