# Git Workflow & Repository Structure Rules

When performing Git operations and managing the repository layout, follow these rules:

## 1. Committing Practices
- **Frequent & Small**: Produce small, logical commits instead of massive monolithic ones.
- **Conventional Commits**: Commit messages must adhere to the Conventional Commits specification (e.g., `feat: ...`, `fix: ...`, `refactor: ...`). 
- **Single Functionality**: Each commit should relate to exactly one piece of functionality or distinct structural change.

## 2. Branching & Merging
- **Branch `dev`**: The `dev` branch is the primary integration branch for pushing and merging code.
- **Branch `main`**: Only stable, finalized releases may be pushed to `main`.
- **Feature Branches**: Organize distinct work into logical feature branches when pushing to the remote server, maintaining an organized Git history.

## 3. Artifact Management & Exclusions
- **LFS Only**: Data files, trained models, and large datasets must **never** be committed directly to Git. Ensure they are tracked with Git Large File Storage (LFS). **CRITICAL**: Every developer and AI agent must have `git-lfs` installed locally for the repository hooks to successfully pull these large files. Run `git lfs install` after cloning or setting up your environment.

## 4. Repository Layout Strictness
Ensure generated code is meticulously placed in the correct directories:
- `src/` for algorithms, network designs, and core agent modules.
- `env/` for MuJoCo wrappers and environment definitions.
- `config/` for experiment configurations (using json, gin, or yaml).
- `experiments/` for executable scripts.
- `docs/` for ReadTheDocs or Doxygen documentation, and decision logs.
- `tests/` for unit tests and verification scripts.
