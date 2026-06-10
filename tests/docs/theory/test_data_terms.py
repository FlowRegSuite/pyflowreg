"""
Executable-documentation tests for docs/theory/data_terms.md.

Each test runs the extracted snippet from ``docs/snippets/theory/data_terms/``
via the ``snippet_runner`` fixture and asserts on the resulting namespace.
"""

import pytest

from pyflowreg.motion_correction import OFOptions

pytestmark = pytest.mark.docs_example


class TestDataTermsSelectDataTerm:
    """docs/theory/data_terms.md, section "Selecting a data term"."""

    def test_select_data_term_executes(self, snippet_runner):
        ns = snippet_runner("theory/data_terms/select_data_term.py")

        options = ns["options"]
        assert isinstance(options, OFOptions)
        # ConstancyAssumption is a str enum; the serialized value is "gc".
        assert options.constancy_assumption == "gc"
        assert options.constancy_assumption.value == "gc"
        # "gc" is the documented default.
        assert OFOptions().constancy_assumption == options.constancy_assumption

    @pytest.mark.parametrize(
        "alias,serialized",
        [
            ("gradient", "gc"),
            ("brightness", "gray"),
            ("census", "cs"),
        ],
    )
    def test_select_data_term_aliases_normalize(self, alias, serialized):
        """The alias table on the page: aliases normalize on validation."""
        options = OFOptions(constancy_assumption=alias)
        assert options.constancy_assumption == serialized
