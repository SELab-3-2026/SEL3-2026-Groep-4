# Contribution Guidelines

This document outlines the contribution protocols for the scientific software engineering project focusing on bio-inspired control architectures for brittle-star-like robots. The primary objective of this project is to produce scientific insight, rather than a commercial product.

## 1. Scientific Context & Methodology

* **Research Focus:** The goal is to study how controller modularity affects learning speed, coordination, and fault tolerance in brittle-star locomotion.
* **Hypothesis-Driven Design:** Clear hypotheses must dictate a structured methodology and rigorous evaluation. All design decisions must be formally documented prior to implementation.
* **Scaffolding Approach:** Development must start with simple setups before progressively increasing the complexity of environments and morphologies.
* **Evaluation of Results:** Negative results possess scientific validity when thoroughly analyzed. If a controller fails to learn locomotion, providing a comprehensive analysis of the failure is considered a strong scientific contribution.
* **Reproducibility:** Contributors must utilize fixed library versions. Configuration systems (such as json, gin, or yaml) must be employed to ensure reproducible runs.

## 2. Clean Code & Code Quality

Code readability is paramount, as code is read far more frequently than it is written.

* **Naming Conventions:** Variables and functions must utilize consistent, intention-revealing names. A long, descriptive name is strictly preferred over a short name accompanied by a comment.
* **Function Design:** Functions must be modular and adhere to the single responsibility principle. Arguments must be minimized, and boolean flag arguments controlling behavior should be avoided.
* **Commenting:** Code must document the "how," while comments are strictly reserved for documenting the "why". Commented-out code is prohibited and must be deleted via version control.
* **YAGNI:** Contributors must adhere to the "You Aren't Gonna Need It" (YAGNI) principle and actively avoid premature optimization.
* **Notebooks:** Jupyter Notebooks are strictly limited to quick prototyping, tutorials, demonstrations, or post-processing analysis. They are explicitly forbidden for general software development because they discourage modularity.

## 3. Version Control & Repository Structure

* **Git Practices:** Commits must be frequent and small. Each commit should relate to exactly one piece of functionality.
* **Branching Strategy:** The `dev` branch serves as the integration branch for pushing and merging code. Only stable releases may be pushed to the `main` branch.
* **Artifact Management:** Data files, trained models, and large datasets must never be committed directly to Git. Git Large File Storage (LFS) must be used for tracking large files. **All developers must have `git-lfs` installed locally** (see `DEVELOPMENT.md` for setup).
* **Repository Layout:** The repository must maintain the following core directories: `src/` for algorithms, `env/` for MuJoCo wrappers, `config/` for experiment configurations, `experiments/` for scripts, `docs/` for Doxygen or ReadTheDocs documentation, and `tests/` for unit tests.

## 4. Architecture & Tooling

* **Algorithms & Frameworks:** Proximal Policy Optimization (PPO) is the recommended baseline algorithm. CleanRL should be used as a starting point and adapted for continuous action spaces. All Artificial Neural Network (ANN) controller architectures must be implemented using Flax.
* **Simulation:** The simulation environment utilizes a MuJoCo brittle star. XML MuJoCo structures must remain realistic and respect morphological constraints.
* **Experiment Tracking:** Weights & Biases (wandb) must be utilized for tracking and logging all experiments.
* **Code Styling:** All code must conform to the chosen style guide (i.e. Google standard). This is enforced using build tools and pre-commit hooks such as flake8, black, or isort.
