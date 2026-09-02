from __future__ import annotations

import subprocess
from pathlib import Path


class ProvisioningError(RuntimeError):
    pass


def clone_repo(repo_url: str, dest_dir: Path, timeout_sec: int = 300) -> Path:
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest_dir)],
        capture_output=True, text=True, timeout=timeout_sec,
    )
    if result.returncode != 0:
        raise ProvisioningError(f"git clone failed:\n{result.stderr}")
    return dest_dir


def install_requirements(repo_dir: Path, timeout_sec: int = 600) -> str:
    """Best-effort dependency install. Returns combined stdout+stderr for
    logging. Two steps:
    1. `pip install -r requirements.txt` if present -- the repo's own deps.
    2. `pip install -e .` if the repo itself is a Python package (setup.py /
       pyproject.toml at root) -- many paper repos (e.g. a fork of
       huggingface/transformers) need this to make `import <package>`
       resolve to the cloned source at all; requirements.txt alone lists
       what the package depends on, not the package itself. This step is
       best-effort/soft-fail (logged, not raised) since editable-install
       failure modes vary a lot across repos and shouldn't hard-stop a run
       that might still be runnable via e.g. PYTHONPATH instead."""
    log = ""
    req_file = repo_dir / "requirements.txt"
    if req_file.exists():
        result = subprocess.run(
            ["pip", "install", "-q", "-r", str(req_file)],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        log += result.stdout + result.stderr
        if result.returncode != 0:
            raise ProvisioningError(log)

    if (repo_dir / "setup.py").exists() or (repo_dir / "pyproject.toml").exists():
        try:
            result = subprocess.run(
                ["pip", "install", "-q", "-e", "."],
                cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout_sec,
            )
            log += result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            log += "\n[editable install timed out, skipped]"
    return log
