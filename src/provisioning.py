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
    """Best-effort dependency install from requirements.txt. Returns combined
    stdout+stderr for logging. No-op (returns "") if no requirements.txt."""
    req_file = repo_dir / "requirements.txt"
    if not req_file.exists():
        return ""
    result = subprocess.run(
        ["pip", "install", "-q", "-r", str(req_file)],
        capture_output=True, text=True, timeout=timeout_sec,
    )
    log = result.stdout + result.stderr
    if result.returncode != 0:
        raise ProvisioningError(log)
    return log
