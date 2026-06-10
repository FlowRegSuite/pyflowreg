# Docs page : docs/user_guide/backends.md ("Available Backends")
# Test      : tests/docs/user_guide/test_backends.py::TestBackendsInspectBackends
# Inputs    : none
# [docs:start]
from pyflowreg.core import list_backends, is_backend_available

backends = list_backends()
print(backends)  # e.g. ['flowreg', 'diso', 'flowreg_torch']
print(is_backend_available("flowreg_cuda"))
# [docs:end]
