"""
Pytest fixtures for executing documentation code examples.

The documentation keeps every runnable code block as a standalone snippet
file under ``docs/snippets/``; the pages render those files via MyST
``literalinclude`` and the tests in ``tests/docs/`` execute them. A test
first calls ``materialize_input`` to create the input files a snippet
references (e.g. ``my_video.tif``) with small synthetic data, then runs
the snippet via ``snippet_runner`` and asserts on the resulting module
namespace.

All synthetic inputs are written through pyflowreg's own video writers
(``get_video_file_writer``), never through raw numpy/h5py/tifffile calls.
"""

import runpy
from pathlib import Path

import numpy as np
import pytest

from pyflowreg.util.io.factory import get_video_file_writer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SNIPPETS_DIR = REPO_ROOT / "docs" / "snippets"

_FORMAT_BY_SUFFIX = {
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".h5": "HDF5",
    ".hdf5": "HDF5",
    ".mat": "MAT",
}


def _synthetic_video(shape, pattern="motion", seed=0):
    """Generate a small synthetic (T, H, W, C) uint16 video.

    ``motion`` moves a bright disc on a circular trajectory over a noisy
    background so the optical flow solver has structure to track;
    ``static`` is constant intensity plus noise.
    """
    T, H, W, C = shape
    rng = np.random.default_rng(seed)
    frames = np.zeros(shape, dtype=np.float32)

    if pattern == "motion":
        y, x = np.ogrid[:H, :W]
        radius = max(2, min(H, W) // 8)
        for t in range(T):
            off_y = int(round(radius * 0.5 * np.sin(2 * np.pi * t / T)))
            off_x = int(round(radius * 0.5 * np.cos(2 * np.pi * t / T)))
            mask = (x - W // 2 - off_x) ** 2 + (y - H // 2 - off_y) ** 2 <= radius**2
            frame = mask.astype(np.float32) * 0.8 + 0.2 * rng.random(
                (H, W), dtype=np.float32
            )
            frames[t] = frame[:, :, None]
    elif pattern == "static":
        frames[:] = 0.5 + 0.05 * rng.random(shape, dtype=np.float32)
    else:
        raise ValueError(f"Unknown synthetic pattern: {pattern!r}")

    frames = np.clip(frames, 0.0, 1.0)
    return (frames * 65535).astype(np.uint16)


def _write_video(path, frames):
    suffix = path.suffix.lower()
    if suffix not in _FORMAT_BY_SUFFIX:
        supported = ", ".join(sorted(_FORMAT_BY_SUFFIX))
        raise ValueError(
            f"Cannot materialize {path.name!r}: unsupported suffix {suffix!r} "
            f"(supported: {supported})"
        )
    writer = get_video_file_writer(str(path), _FORMAT_BY_SUFFIX[suffix])
    try:
        writer.write_frames(frames)
    finally:
        writer.close()


@pytest.fixture
def materialize_input(tmp_path, monkeypatch):
    """Create the input file(s) a documentation snippet references.

    Returns ``make(filename, shape=(12, 32, 48, 2), pattern="motion")``,
    which writes a synthetic video under ``tmp_path`` with the exact
    filename the snippet uses (so the published example never has to be
    edited to point at test fixtures) and returns the path. Also changes
    the working directory to ``tmp_path`` so relative filenames resolve.
    """
    monkeypatch.chdir(tmp_path)

    def make(filename, shape=(12, 32, 48, 2), pattern="motion", seed=0):
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_video(path, _synthetic_video(shape, pattern=pattern, seed=seed))
        return path

    return make


@pytest.fixture
def materialize_session(tmp_path, monkeypatch):
    """Create a tiny multi-recording session directory for session snippets.

    Returns ``make(n_recordings=3, filename_pattern="recording_{:03d}.tif",
    shape=(6, 64, 64, 1), root="session")``, which writes ``n_recordings``
    synthetic recordings of the same textured scene with a small constant
    integer shift per recording (the misalignment the session pipeline
    corrects) and returns the session root path. The working directory is
    changed to ``tmp_path``.
    """
    monkeypatch.chdir(tmp_path)

    def make(
        n_recordings=3,
        filename_pattern="recording_{:03d}.tif",
        shape=(6, 64, 64, 1),
        root="session",
    ):
        T, H, W, C = shape
        rng = np.random.default_rng(7)
        session_root = tmp_path / root
        session_root.mkdir(parents=True, exist_ok=True)

        # Textured base frame: noisy background, one Gaussian blob, two
        # rectangles, so cross-correlation and flow have features to lock on.
        base = rng.random((H, W), dtype=np.float32) * 0.2
        y, x = np.ogrid[:H, :W]
        base += np.exp(-((y - H // 2) ** 2 + (x - W // 2) ** 2) / (0.02 * H * W))
        base[H // 4 : H // 2, W // 4 : W // 2] += 0.5
        base[int(H * 0.6) : int(H * 0.8), int(W * 0.55) : int(W * 0.75)] += 0.4

        max_shift = max(1, min(H, W) // 16)
        for i in range(n_recordings):
            dy = int(round(max_shift * np.sin(2 * np.pi * i / max(n_recordings, 2))))
            dx = int(round(max_shift * np.cos(2 * np.pi * i / max(n_recordings, 2))))
            shifted = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
            frames = np.zeros(shape, dtype=np.float32)
            for t in range(T):
                noisy = shifted + 0.05 * rng.random((H, W), dtype=np.float32)
                frames[t] = noisy[:, :, None]
            frames = np.clip(frames, 0.0, 1.0)
            _write_video(
                session_root / filename_pattern.format(i),
                (frames * 65535).astype(np.uint16),
            )

        return session_root

    return make


@pytest.fixture
def snippet_runner(tmp_path, monkeypatch):
    """Execute a documentation snippet in an isolated working directory.

    Returns ``run(snippet_relpath)``, which executes
    ``docs/snippets/<snippet_relpath>`` with :func:`runpy.run_path` after
    changing the working directory to ``tmp_path`` (where
    ``materialize_input`` placed the snippet's input files) and returns
    the module namespace, so tests can assert on the variables the
    published example creates (e.g. ``ns["registered"].shape``).
    """
    monkeypatch.chdir(tmp_path)

    def run(snippet_relpath):
        snippet_path = SNIPPETS_DIR / snippet_relpath
        if not snippet_path.exists():
            raise FileNotFoundError(f"Documentation snippet not found: {snippet_path}")
        return runpy.run_path(str(snippet_path))

    return run
