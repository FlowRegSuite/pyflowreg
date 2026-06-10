# Docs page : docs/user_guide/parallelization.md ("Executor Registration")
# Test      : tests/docs/user_guide/test_parallelization.py::TestParallelizationExecutorRegistry
# Inputs    : none
# [docs:start]
from pyflowreg._runtime import RuntimeContext

# Check available executors
available = RuntimeContext.get("available_parallelization", set())
print(f"Available executors: {sorted(available)}")
# [docs:end]
