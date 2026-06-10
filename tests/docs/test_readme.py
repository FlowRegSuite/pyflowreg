"""
Executes every ```python fenced code block in README.md.

The blocks are extracted from the README at collection time with a regex
(no snippet files are maintained for the README) and each block is
executed in an isolated namespace with the working directory set to a
temporary path.

README.md currently contains no python fences (only bash/powershell and
a BibTeX block), so the parametrized test collects an empty parameter
set and the extraction test skips visibly with
"README has no python code blocks". If python examples are added to the
README later, they are picked up and executed automatically.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.docs_example

# tests/docs/test_readme.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"

_PYTHON_FENCE_RE = re.compile(r"^```python[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _extract_python_blocks():
    """Return the contents of all ```python fenced blocks in README.md."""
    text = README_PATH.read_text(encoding="utf-8")
    return [match.group(1) for match in _PYTHON_FENCE_RE.finditer(text)]


_README_PYTHON_BLOCKS = _extract_python_blocks()


class TestReadmePythonBlocks:
    """Execute each python code block found in README.md."""

    def test_readme_python_block_extraction(self):
        """Extraction ran against the real README; skip visibly if empty."""
        assert README_PATH.exists()
        text = README_PATH.read_text(encoding="utf-8")
        # The README does contain fenced blocks (bash etc.), so an empty
        # result means "no python fences", not a broken regex.
        assert "```" in text
        blocks = _extract_python_blocks()
        assert blocks == _README_PYTHON_BLOCKS
        if not blocks:
            pytest.skip("README has no python code blocks")

    # With zero blocks the parameter set is empty and pytest reports a
    # single skipped test ("empty parameter set"), keeping the gap visible.
    @pytest.mark.parametrize(
        "block_index",
        range(len(_README_PYTHON_BLOCKS)),
        ids=[f"block{i}" for i in range(len(_README_PYTHON_BLOCKS))],
    )
    def test_readme_python_block_executes(self, block_index, tmp_path, monkeypatch):
        """Each README python block runs as-is in a tmp working directory."""
        monkeypatch.chdir(tmp_path)
        code = _README_PYTHON_BLOCKS[block_index]
        namespace = {"__name__": "__readme__"}
        exec(
            compile(code, f"README.md[python block {block_index}]", "exec"),
            namespace,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
