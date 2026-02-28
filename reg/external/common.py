from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Sequence


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary '{name}' not found on PATH")
    return path


def run(cmd: Sequence[str], *, cwd: str | Path | None = None) -> None:
    subprocess.run(list(cmd), cwd=str(cwd) if cwd is not None else None, check=True)

