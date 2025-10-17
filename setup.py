"""
Minimal setup.py for backwards compatibility and editable installs.
All configuration is in pyproject.toml.
"""

from setuptools import setup

# This enables:
# - pip install .
# - pip install -e .
# - python setup.py develop (deprecated but still used)
#
# All actual configuration comes from pyproject.toml
setup()
