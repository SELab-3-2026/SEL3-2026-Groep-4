from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import wandb
import argparse

# tune these depending on network / W&B limits
MAX_RUN_WORKERS = 8
MAX_FILE_WORKERS = 16

api = wandb.Api()

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def download_file(file, run_dir):
    target = run_dir / file.name
    try:
        # skip existing files
        if target.exists():
            return f"SKIP {target}"

        target.parent.mkdir(parents=True, exist_ok=True)

        file.download(root=run_dir, replace=False)

        return f"DONE {target}"

    except Exception as e:
        return f"FAIL {target}: {e}"


def download_run(run, root):
    run_dir = root / f"{run.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    safe_print(f"\n=== {run.name} ({run.id}) ===")

    files = list(run.files())

    with ThreadPoolExecutor(max_workers=MAX_FILE_WORKERS) as executor:
        futures = [executor.submit(download_file, file, run_dir) for file in files]

        for future in as_completed(futures):
            safe_print(future.result())

    # OPTIONAL: download artifacts too
    # for artifact in run.logged_artifacts():
    #     artifact_dir = run_dir / "artifacts" / artifact.name
    #     artifact.download(root=artifact_dir)

    safe_print(f"Finished {run.name}")


def main(entity: str, project: str, root: Path):
    root.mkdir(exist_ok=True)

    runs = list(api.runs(f"{entity}/{project}"))

    safe_print(f"Found {len(runs)} runs")

    with ThreadPoolExecutor(max_workers=MAX_RUN_WORKERS) as executor:
        futures = [executor.submit(download_run, run, root) for run in runs]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                safe_print("RUN FAILED:", e)

    safe_print("\nAll downloads complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="SEL3-2026-Groep-4")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--root", type=str, default="runs")
    args = parser.parse_args()

    root = Path(args.root)
    main(entity=args.entity, project=args.project, root=root)
