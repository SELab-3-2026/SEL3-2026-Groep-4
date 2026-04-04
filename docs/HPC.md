# HPC Guide

Full documentation: <https://docs.hpc.ugent.be/>

## Storage Overview

- **Run Outputs**: Written to `$VSC_SCRATCH` during the job (fast I/O) and copied to `$VSC_DATA` at the end for persistence.
- **Virtual Environments**: Managed on **`$VSC_DATA`** by mirroring configuration files. This avoids the 3GB home quota without requiring symlinks in the project root.

## Initial Environment Setup

Run **once** after cloning the repository. Ensure you are logged into a **compute node** on `donphan` or `joltik`. 

> [!IMPORTANT]
> To avoid the **3GB home directory quota limit**, the installation script mirrors your configuration files to **`$VSC_DATA`** (25GB+ quota). The `vsc-venv` tool then automatically creates and manages the environment on the larger partition.

```bash
# 1. Start an interactive session (donphan for debug, joltik for training)
qsub -I -l nodes=1:ppn=8:gpus=1

# 2. Run the streamlined install script
cd "${PBS_O_WORKDIR}"
bash scripts/hpc/install.sh
```

## Interactive Debugging

You can use the same `install.sh` script to quickly activate your environment for interactive work.

```bash
# Request an interactive job
qsub -I -l nodes=1:ppn=4 -l walltime=1:00:00

# Change to project directory and run install.sh to sync and activate
cd "$PBS_O_WORKDIR"
bash scripts/hpc/install.sh
```

### Verification Commands

After installation, run these commands to ensure your environment is set up correctly:

1. **Verify Location**:
   ```bash
   # Confirm that NO 'venvs' folder appeared in your project root
   ls -d venvs 2>/dev/null  # Should return 'not found'
   
   # Confirm the environment is on the data partition
   python -c "import torch; print(torch.__file__)"
   # Expected: /kyukon/data/gent/vsc... or similar
   ```

2. **Verify GPU Access**:
   ```bash
   python -c "import torch; import jax; print(f'Torch CUDA: {torch.cuda.is_available()}'); print(f'JAX Devices: {jax.devices()}')"
   ```
   *Expected output: `Torch CUDA: True` and `JAX Devices: [CudaDevice(id=0)]`.*

3. **Verify Home Quota**:
   ```bash
   df -h ~ # Should show low usage (< 1GB typically)
   ```

## Submitting Batch Training Jobs

```bash
# Submit to the default cluster (joltik)
qsub scripts/hpc/train.pbs

# To choose a different cluster (e.g. donphan debug) without touching code
module swap cluster/donphan && qsub scripts/hpc/train.pbs
```

The `train.pbs` script automatically handles its own activation using the mirrored configurations on `$VSC_DATA`.

## Managing Dependencies

`env/hpc/requirements.txt` is auto-generated from `pyproject.toml`. To regenerate:

```bash
uv run scripts/export_hpc_requirements.py
```

Modules listed in `env/hpc/modules.txt` are automatically excluded from the pip requirements to save space and use HPC-optimized binaries.
