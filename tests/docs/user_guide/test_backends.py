"""
Tests for the code examples in docs/user_guide/backends.md.

Executes the snippets from ``docs/snippets/user_guide/backends/`` via the
``snippet_runner`` fixture and asserts on the resulting module namespace.
"""

import pytest

pytestmark = pytest.mark.docs_example


class TestBackendsInspectBackends:
    """docs/user_guide/backends.md -- "Available Backends" registry introspection."""

    def test_inspect_backends_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/backends/inspect_backends.py")

        backends = ns["backends"]
        assert isinstance(backends, list)
        # The default backend is always registered when pyflowreg.core imports.
        assert "flowreg" in backends
