---
name: Lint and Format Code
description: Instructions for checking code style and formatting using Ruff.
---

# Skill: Lint and Format Code

The goal of this skill is to enforce the project's adherence to the Google style standard and to maintain high code quality across the python source files.

## Instructions

1. **Format Code**: To automatically format all Python files according to the `ruff.toml` specifications:
   ```bash
   uv run ruff format .
   ```

2. **Check for Lints / Auto-fix**: To check the repository for style violations and automatically fix safe corrections:
   ```bash
   uv run ruff check --fix .
   ```

## Important Considerations
- The `ruff.toml` file at the root handles all lint and format configuration. Do not ignore configurations when applying fixes.
- If Ruff points out complex errors that cannot be auto-fixed, analyze the code and manually address the violations, prioritizing descriptive naming and adherence to the single-responsibility principle.
