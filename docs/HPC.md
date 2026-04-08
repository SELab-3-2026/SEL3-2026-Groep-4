# HPC Guide

Full documentation: <https://docs.hpc.ugent.be/>

## Storage Overview

- **Run Outputs**: Written to `$VSC_SCRATCH` during the job (fast I/O) and copied to `$VSC_DATA` at the end for persistence.
- **Virtual Environments**: Managed on **`$VSC_DATA`** by mirroring configuration files. This avoids the 3GB home quota without requiring symlinks in the project root.

## Initial Environment Setup

Run **once** after cloning the repository. This script handles all modules, mirroring, and environment synchronization.

```bash
# Option A: Interactive (on a compute node)
module swap cluster/donphan  # Debug cluster (CPU only)
# OR for GPU clusters:
# module swap cluster/joltik
# module swap cluster/accelgor
# module swap cluster/litleo

qsub -I -l nodes=1:gpus=1  # Only for GPU clusters
cd "${PBS_O_WORKDIR}"
bash scripts/hpc/install.sh

# Option B: Batch (Run in background)
# NOTE: GPU clusters (joltik/accelgor/litleo) require -l gpus=1 at runtime
qsub -l gpus=1 scripts/hpc/install.sh
```

## Production vs. Debug Clusters

Our scripts are cluster-agnostic and do **not** have hardcoded GPU requirements. Instead, you must request GPUs at runtime using the `-l gpus=1` flag when submitting to a production GPU cluster.

### Debugging (Donphan)
The `donphan` cluster does not support GPUs. Simply run the scripts without extra resource flags:
```bash
module swap cluster/donphan
qsub scripts/hpc/train.pbs
```

### Production (Joltik, Accelgor, Litleo)
These clusters provide GPU acceleration and **require** a GPU request at runtime:
```bash
module swap cluster/joltik  # or accelgor/litleo
qsub -l gpus=1 scripts/hpc/train.pbs
```

## Interactive Debugging

To activate your environment for interactive work, simply run the same `install.sh` script.

```bash
qsub -I -l nodes=1:ppn=4 -l walltime=1:00:00
cd "$PBS_O_WORKDIR"
bash scripts/hpc/install.sh
```

### Verification Commands

After installation, run these commands to ensure your environment is set up correctly:

1. **Verify Quota Safety**:
   ```bash
   ls -d venvs 2>/dev/null && echo "FAIL" || echo ">>> PASS: Project root is clean."
   ```
2. **Verify Library Versions (NumPy Fix)**:
   ```bash
   python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
   # Expected: 2.x.x (Venv version), not 1.2x (System version)
   ```
3. **Verify GPU Access**:
   ```bash
   python -c "import torch, jax; print(f'GPU: {torch.cuda.is_available()}'); print(f'JAX: {jax.devices()}')"
   ```

## Managing Dependencies

`env/hpc/requirements.txt` is auto-generated from `pyproject.toml`. To regenerate:

```bash
uv run scripts/hpc/export_requirements.py
```

Modules listed in `env/hpc/modules.txt` are automatically excluded from the pip requirements to save space and use HPC-optimized binaries.
