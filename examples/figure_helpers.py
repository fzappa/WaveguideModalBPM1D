"""Shared helpers for the example scripts in `examples/` (save PNGs, silence stdout)."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
FIGURES_DIR = EXAMPLES_DIR / "figures"


def ensure_repo_on_path() -> None:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


@contextlib.contextmanager
def silent_stdout() -> contextlib.AbstractContextManager[None]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def save_figure(fig, png_name: str, *, dpi: int = 120) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / png_name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return path
