# HPC Guide

Full documentation: <https://docs.hpc.ugent.be/>

## Cluster Selection

Choose the appropriate cluster before submitting a job with `module swap cluster/<name>`. The default login cluster is **doduo**.

| Cluster   | Type                    | Use case                                      |
|-----------|-------------------------|-----------------------------------------------|
| `donphan` | Interactive / debug GPU | First-time setup, interactive debugging       |
| `doduo`   | CPU (default login)     | Rapid iteration, CPU-only smoke tests         |
| `joltik`  | GPU (A100 ¼-slice)      | Standard training runs                        |
| `accelgor`| GPU (A100 full)         | Large-scale / long experiments                |
| `litleo`  | GPU                     | Alternative GPU option                        |

> **Rule:** use at most **1 GPU per group at a time** on shared GPU clusters.  
> Check the current queue load at <https://shieldon.ugent.be:8083/pbsmon-web-users/>.

---

## Storage Overview

The HPC provides three filesystems for different purposes. Understanding this is critical to avoid filling up your home directory.

| Variable        | Typical size | Purpose                                              |
|-----------------|-------------|------------------------------------------------------|
| `$VSC_HOME`     | ~3 GB       | Config files, SSH keys, project source code only     |
| `$VSC_DATA`     | ~25 GB      | Persistent outputs: trained models, final logs       |
| `$VSC_SCRATCH`  | Large       | Fast I/O during jobs: caches, intermediate files     |

**Important:**
- Clone the repository into `$VSC_HOME` — it is small in size and accessible from all clusters.
- All caches (pip, uv, matplotlib) must be redirected to `$VSC_SCRATCH` to avoid filling `$VSC_HOME`.
- Run outputs are written to `$VSC_SCRATCH` during the job (fast I/O) and copied to `$VSC_DATA` at the end for persistence.
- `$VSC_SCRATCH` may be purged periodically — do not use it as long-term storage.

Check your quota: <https://account.vscentrum.be> (Usage section).

---

## Initial Environment Setup

Run **once** from a login shell on `donphan` after cloning the repository:

```bash
# 1. Connect via the web portal → HPC Login → Interactive Apps > Shell (tmux)
#    Set cluster to "donphan (interactive/debug)"

# 2. Clone the project into $VSC_HOME (if not done yet)
cd $VSC_HOME
git clone <repo-url>
cd 2026SEL3-project-BrittleStar

# 3. Run the install script
bash scripts/hpc/install.sh
```

This uses the official [`vsc-venv`](https://docs.hpc.ugent.be/Linux/setting_up_python_virtual_environments/#vsc-venv-python-virtual-environment-wrapper-script) wrapper to:
- Redirect caches to `$VSC_SCRATCH` (to preserve your `$VSC_HOME` quota)
- Load the EasyBuild modules listed in `env/hpc/modules.txt` (JAX, Flax, WandB, …)
- Create a per-cluster virtual environment in `$VSC_DATA`
- Pip-install the remaining packages from `env/hpc/requirements.txt`
- Register a Jupyter kernel named `SEL3 (<cluster>)`

> **Note:** virtual environments are cluster-specific. Re-run the script when switching to a new cluster.

---

## Interactive Debugging on donphan

The `donphan` cluster provides quick access and is ideal for verifying your environment before submitting batch jobs.

### Option A — Interactive shell session

```bash
# Swap to the debug cluster (from any login node)
module swap cluster/donphan

# Request an interactive job (1 node, 4 cores)
qsub -I -l nodes=1:ppn=4 -l walltime=1:00:00

# Once inside the job — redirect caches to scratch first
export PIP_CACHE_DIR="$VSC_SCRATCH/.cache/pip"
export UV_CACHE_DIR="$VSC_SCRATCH/.cache/uv"

# Change to the project directory and activate environment
cd "$PBS_O_WORKDIR"
module load vsc-venv
source vsc-venv --activate --modules env/hpc/modules.txt

# Set headless rendering backend
export MUJOCO_GL=egl

# Run the smoke test
python src/train.py --config configs/hpc/smoke_test.yaml
```

### Option B — JupyterLab session (HPC web portal)

1. In the web portal go to **Interactive Apps → JupyterLab RHEL9**
2. Set the following options:

   | Option              | Value                                           |
   |---------------------|-------------------------------------------------|
   | Cluster             | `donphan (interactive/debug)`                   |
   | Number of nodes     | 1                                               |
   | Number of cores     | 4                                               |
   | JupyterLab version  | `4.2.5 GCCcore-13.3.0`                          |
   | Custom code         | *(leave blank — vsc-venv handles modules)*      |

3. Click **Launch**, wait for the session to start, then **Connect**.
4. In JupyterLab, select the kernel **`SEL3 (<cluster>)`**.
5. Verify GPU access:

   ```python
   import jax
   print(jax.default_backend())  # expected: 'gpu'
   print(jax.devices())          # expected: [CudaDevice(id=0)]
   ```

> **Warning:** JAX can only be loaded by one kernel at a time. Shut down other kernels before switching notebooks.

---

## Submitting Batch Training Jobs

```bash
# Default cluster (joltik — one A100 GPU slice)
qsub scripts/hpc/train.pbs

# Switch to a different GPU cluster first
module swap cluster/accelgor
qsub scripts/hpc/train.pbs
```

The job script automatically:
- Redirects all caches to `$VSC_SCRATCH`
- Writes run outputs to `$VSC_SCRATCH/runs/<job_id>` during the run
- Copies final results to `$VSC_DATA/runs/<job_id>` on completion
- Writes PBS stdout/stderr to `$VSC_DATA/runs/<job_id>/job.out` / `job.err`

Monitor your jobs:

```bash
qstat          # list your jobs
qstat -f <id>  # detailed info for a specific job
qdel <id>      # cancel a job
```

---

## Managing Dependencies

`env/hpc/requirements.txt` is auto-generated by CI whenever `pyproject.toml` changes. To regenerate locally:

```bash
uv run scripts/export_hpc_requirements.py
```

Do **not** edit `env/hpc/requirements.txt` by hand — edit `pyproject.toml` instead.
