"""
Drift guards for the documentation single-source example mechanism.

Runnable documentation examples live as snippet files under docs/snippets/,
are rendered into pages via MyST literalinclude, and are executed by the
tests in tests/docs/. These guards enforce the three legs of that contract
with pure text parsing (fast, no numerics):

1. No raw ```python fence appears in docs pages unless it is listed in
   ``ALLOWED_INLINE_PYTHON`` (illustrative-only blocks: GPU/optional-dep
   examples, pseudo-code fragments, import-only fragments covered by
   dedicated tests). The allowlist is exact (page + first line + count),
   so converting a block to a snippet requires shrinking the list — it can
   only burn down, never silently grow.
2. Every snippet file is literalinclude'd from exactly one docs page, and
   every literalinclude target exists.
3. Every snippet file is referenced by name in at least one test module
   under tests/docs/.
"""

import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SNIPPETS_DIR = DOCS_DIR / "snippets"
TESTS_DOCS_DIR = Path(__file__).resolve().parent

_PYTHON_FENCE = re.compile(r"^```python[ \t]*\r?\n(.*?)^```", re.M | re.S)
_LITERALINCLUDE = re.compile(r"^```\{literalinclude\}[ \t]+(\S+)", re.M)

# (page -> first non-empty line of the allowed inline block -> count).
# Reasons, summarized per page:
# - installation.md: install-verification import; exercised by the suite.
# - api/motion_correction.md: abridged signature sketch (autodoc below it).
# - api/session.md: import-only/loader fragments (covered by
#   tests/docs/api/test_session.py), TOML/GPU sketches, REPL illustration.
# - 3d_volumes.md: long-running full-pipeline and napari visualization.
# - backends.md: optional-dep (torch/cv2) examples; registry sketch with
#   process-wide side effects.
# - configuration.md: GPU backend blocks (torch + CUDA).
# - file_formats.md: OFOptions fragments without imports, illustrating one
#   format string each; MDF needs the Windows COM server.
# - multi_session.md: fragments mutating an undefined `config`,
#   visualization, external-tool (CaImAn/Suite2p) sketches.
# - online_processing.md: alternative-call-form fragments of the
#   streaming example (tested via the extracted streaming_loop snippet).
# - parallelization.md: executor/performance fragments without imports;
#   multiprocessing full runs (spawn cost); GPU/diso optional deps;
#   dataclass display.
# - workflows.md: multiprocessing full run, illustrative streaming/napari/
#   threading sketches, callback signature documentation.
ALLOWED_INLINE_PYTHON = {
    "docs/installation.md": {
        "import pyflowreg": 1,
    },
    "docs/api/motion_correction.md": {
        "compensate_arr(": 1,
    },
    "docs/api/session.md": {
        "from pyflowreg.session.config import SessionConfig": 1,
        'config = SessionConfig.from_toml("session.toml")': 1,
        'config = SessionConfig.from_yaml("session.yml")': 1,
        'config = SessionConfig.from_file("session.toml")  # or .yml/.yaml': 1,
        "from pyflowreg.session.stage1_compensate import run_stage1": 1,
        "from pyflowreg.session.stage1_compensate import run_stage1_array": 1,
        "from pyflowreg.session.stage2_between_avgs import run_stage2": 1,
        "config = SessionConfig(": 1,
        "from pyflowreg.session.stage3_valid_mask import run_stage3": 1,
        "import numpy as np": 1,
        "from pyflowreg.core.warping import (": 1,
        "from pyflowreg.session.config import get_array_task_id": 1,
        "from pyflowreg.session.stage1_compensate import atomic_save_npy, atomic_save_npz": 1,
        "# Safe numpy array save": 1,
    },
    "docs/user_guide/3d_volumes.md": {
        "from pyflowreg.motion_correction import compensate_recording, OFOptions": 1,
        "import numpy as np": 2,
    },
    "docs/user_guide/backends.md": {
        "from pyflowreg.motion_correction import OFOptions, compensate_recording": 1,
        "import cv2": 1,
        "import numpy as np": 1,
    },
    "docs/user_guide/configuration.md": {
        "from pyflowreg.motion_correction import compensate_recording, OFOptions": 1,
        "# Automatic device selection (CUDA if available, otherwise CPU)": 1,
    },
    "docs/user_guide/file_formats.md": {
        "options = OFOptions(": 10,
        "# Larger buffers read more frames per batch and use more memory;": 1,
    },
    "docs/user_guide/multi_session.md": {
        "import numpy as np": 3,
        'config.flow_options = {"buffer_size": 500}': 1,
        'config.pattern = "*_001.tif"  # Test with first of each condition': 1,
        "import matplotlib.pyplot as plt": 1,
        "for w in displacement_fields:": 1,
        "import os": 1,
        "# For 16GB RAM": 1,
    },
    "docs/user_guide/online_processing.md": {
        "# From an explicit frame stack (T, H, W, C): frames are preregistered first": 1,
        "registered, flow = flow_reg(frame)": 1,
    },
    "docs/user_guide/parallelization.md": {
        "# Auto-selection preference: multiprocessing -> threading -> sequential": 1,
        'config = RegistrationConfig(parallelization="threading")  # Force threading': 1,
        'options = OFOptions(flow_backend="flowreg")': 1,
        'options = OFOptions(flow_backend="diso")': 1,
        "from pyflowreg.motion_correction import OFOptions": 1,
        'options = OFOptions(flow_backend="flowreg_cuda")': 1,
        "@dataclass": 1,
        "config = RegistrationConfig(": 1,
        "from pyflowreg.motion_correction import compensate_recording, OFOptions": 1,
        "# These are equivalent:": 1,
        "# Reduce memory usage": 1,
        "# Maximize throughput": 1,
        "import os": 1,
        "# Multiprocessing with large buffers": 1,
        "# Threading to avoid I/O contention": 1,
        "# Quick preview with minimal resources": 1,
        "# Maximum performance for production": 1,
        "# Optimize for limited RAM": 1,
        "# Low-memory configuration": 1,
        "# Fast processing configuration": 1,
        "# Debug configuration": 1,
    },
    "docs/user_guide/workflows.md": {
        "from pyflowreg.motion_correction import compensate_recording, OFOptions": 1,
        "import numpy as np": 1,
        "# Set a new reference from a (T, H, W, C) stack;": 1,
        "import napari": 1,
        "import threading": 1,
        "# Progress callback": 1,
    },
}


def _docs_pages():
    for page in sorted(DOCS_DIR.rglob("*.md")):
        parts = page.relative_to(DOCS_DIR).parts
        if parts[0] in ("_build", "snippets"):
            continue
        yield page


def _page_key(page):
    return "docs/" + page.relative_to(DOCS_DIR).as_posix()


def _inline_python_blocks():
    found = Counter()
    for page in _docs_pages():
        text = page.read_text(encoding="utf-8")
        for match in _PYTHON_FENCE.finditer(text):
            lines = (ln.strip() for ln in match.group(1).splitlines())
            first = next((ln for ln in lines if ln), "")
            found[(_page_key(page), first)] += 1
    return found


def _allowed_counter():
    return Counter(
        {
            (page, first): count
            for page, entries in ALLOWED_INLINE_PYTHON.items()
            for first, count in entries.items()
        }
    )


class TestInlinePythonAllowlist:
    """Leg 1: raw python fences in docs must be on the explicit allowlist."""

    def test_no_unlisted_inline_python_blocks(self):
        found = _inline_python_blocks()
        allowed = _allowed_counter()
        unlisted = {
            key: count for key, count in found.items() if count > allowed.get(key, 0)
        }
        assert not unlisted, (
            "Raw ```python blocks found in docs pages that are not on the "
            "allowlist. Either convert them to docs/snippets/<page>/<name>.py "
            "+ literalinclude + a test in tests/docs/, or add them to "
            f"ALLOWED_INLINE_PYTHON with a reason. Offenders: {unlisted}"
        )

    def test_allowlist_has_no_stale_entries(self):
        found = _inline_python_blocks()
        allowed = _allowed_counter()
        stale = {
            key: count for key, count in allowed.items() if found.get(key, 0) < count
        }
        assert not stale, (
            "ALLOWED_INLINE_PYTHON lists blocks that no longer exist (or exist "
            "fewer times) in the docs — remove them so the allowlist only "
            f"burns down: {stale}"
        )


class TestSnippetLinkage:
    """Legs 2 and 3: snippet <-> page <-> test wiring."""

    def test_snippets_exist(self):
        snippets = list(SNIPPETS_DIR.rglob("*.py"))
        assert snippets, f"No snippet files found under {SNIPPETS_DIR}"

    def test_every_snippet_has_docs_markers(self):
        missing = []
        for snippet in sorted(SNIPPETS_DIR.rglob("*.py")):
            text = snippet.read_text(encoding="utf-8")
            if "[docs:start]" not in text or "[docs:end]" not in text:
                missing.append(snippet.relative_to(SNIPPETS_DIR).as_posix())
        assert not missing, f"Snippets missing [docs:start]/[docs:end]: {missing}"

    def test_every_snippet_is_literalincluded_exactly_once(self):
        refs = Counter()
        broken = []
        for page in _docs_pages():
            text = page.read_text(encoding="utf-8")
            for match in _LITERALINCLUDE.finditer(text):
                target = (page.parent / match.group(1)).resolve()
                if not target.exists():
                    broken.append(f"{_page_key(page)} -> {match.group(1)}")
                else:
                    refs[target] += 1
        assert not broken, f"literalinclude targets that do not exist: {broken}"

        problems = []
        for snippet in sorted(SNIPPETS_DIR.rglob("*.py")):
            count = refs.get(snippet.resolve(), 0)
            if count != 1:
                rel = snippet.relative_to(SNIPPETS_DIR).as_posix()
                problems.append(f"{rel}: referenced {count} times (expected 1)")
        assert not problems, (
            "Each snippet must be literalinclude'd from exactly one docs page: "
            f"{problems}"
        )

    def test_every_snippet_has_a_test(self):
        corpus = "\n".join(
            p.read_text(encoding="utf-8")
            for p in sorted(TESTS_DOCS_DIR.rglob("test_*.py"))
            if p.name != "test_snippet_coverage.py"
        )
        missing = []
        for snippet in sorted(SNIPPETS_DIR.rglob("*.py")):
            rel = snippet.relative_to(SNIPPETS_DIR).as_posix()
            # Accept either the full relative path as a literal, or the
            # directory + filename appearing separately (test modules may
            # factor the directory into a SNIPPET_DIR constant used in
            # f-strings).
            parent = snippet.parent.relative_to(SNIPPETS_DIR).as_posix()
            if rel in corpus or (parent in corpus and snippet.name in corpus):
                continue
            missing.append(rel)
        assert not missing, (
            "Snippets not referenced by any test under tests/docs/ "
            f"(snippet_runner path strings use forward slashes): {missing}"
        )
