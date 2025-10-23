"""
Runtime context management for PyFlowReg.
Handles feature detection, parallelization modes, and configuration.
"""

import os
import sys
import contextvars
import json
import warnings
from typing import Any, Dict, Optional, Set, Type, Union
from contextlib import contextmanager
from importlib import import_module


class RuntimeContext:
    """
    Global runtime configuration with thread-local overrides.
    Manages available features, parallelization modes, and runtime settings.
    """

    # Global configuration (survives pickling for multiprocessing)
    _config: Dict[str, Any] = {
        "max_workers": os.cpu_count(),
        "available_features": set(),
        "available_parallelization": set(),
        "available_backends": set(),
        "parallelization_registry": {},
    }

    # Context-local overrides (supports async and threading)
    _context_overrides: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
        "runtime_overrides", default={}
    )

    # Track if initialization has been done
    _initialized = False

    @classmethod
    def init(cls, force: bool = False) -> None:
        """
        Auto-detect available features, backends, and parallelization modes.
        Called automatically on first access or can be forced.

        Args:
            force: Force re-initialization even if already initialized
        """
        if cls._initialized and not force:
            return

        cls._detect_backends()
        cls._detect_parallelization_modes()
        cls._detect_optional_features()
        cls._initialized = True

    @classmethod
    def _detect_backends(cls) -> None:
        """Detect available optical flow backends based on installed packages."""
        # Variational OF is always available (built-in)
        cls._config["available_backends"].add("variational")

        # Check for RAFT-2P (deep learning backend)
        try:
            import_module("raft_2p")
            cls._config["available_backends"].add("raft-2p")
        except ImportError:
            pass

        # Check for other potential backends
        backend_checks = []

        for backend_name, module_name in backend_checks:
            try:
                import_module(module_name)
                cls._config["available_backends"].add(backend_name)
            except ImportError:
                pass

    @classmethod
    def _detect_parallelization_modes(cls) -> None:
        """Detect available parallelization modes based on system capabilities."""
        # Sequential is always available
        cls._config["available_parallelization"].add("sequential")

        # Threading is always available in Python
        cls._config["available_parallelization"].add("threading")

        # Multiprocessing availability depends on platform
        if hasattr(os, "fork"):
            # Unix-like systems with fork support
            cls._config["available_parallelization"].add("multiprocessing")
            cls._config["available_parallelization"].add("multiprocessing_fork")

        # Spawn method is available on all platforms
        if sys.platform == "win32" or sys.version_info >= (3, 8):
            cls._config["available_parallelization"].add("multiprocessing_spawn")
            cls._config["available_parallelization"].add("multiprocessing")

        # Check for distributed computing libraries
        try:
            import_module("dask")
            cls._config["available_parallelization"].add("dask")
        except ImportError:
            pass

        try:
            import_module("ray")
            cls._config["available_parallelization"].add("ray")
        except ImportError:
            pass

    @classmethod
    def _detect_optional_features(cls) -> None:
        """Detect other optional features and accelerators."""
        # GPU support via CuPy
        try:
            warnings.filterwarnings(
                "ignore",
                message="CUDA path could not be detected",
                category=UserWarning,
                module=r"cupy\._environment",
            )
            cp = import_module("cupy")
            if cp.cuda.runtime.getDeviceCount() > 0:
                cls._config["available_features"].add("gpu_cupy")
        except (ImportError, Exception):
            pass

        # GPU support via PyTorch
        try:
            torch = import_module("torch")
            if torch.cuda.is_available():
                cls._config["available_features"].add("gpu_torch")
        except ImportError:
            pass

        # GPU support via JAX
        try:
            jax = import_module("jax")
            if len(jax.devices("gpu")) > 0:
                cls._config["available_features"].add("gpu_jax")
        except ImportError:
            pass

        # Intel MKL acceleration
        try:
            import_module("mkl")
            cls._config["available_features"].add("mkl")
        except ImportError:
            pass

        # OpenCV acceleration
        try:
            import_module("cv2")
            cls._config["available_features"].add("opencv")
        except ImportError:
            pass

    @classmethod
    def register_parallelization_executor(
        cls, name: str, executor_class: Union[Type, str]
    ) -> None:
        """
        Register a parallelization executor class.

        Args:
            name: Name of the parallelization mode
            executor_class: Class implementing the executor interface or dotted path string
        """
        # Store as dotted path string for pickle compatibility
        if isinstance(executor_class, type):
            module = executor_class.__module__
            class_name = executor_class.__qualname__
            dotted_path = f"{module}.{class_name}"
        else:
            dotted_path = executor_class

        cls._config["parallelization_registry"][name] = dotted_path
        cls._config["available_parallelization"].add(name)

    @classmethod
    def get_parallelization_executor(cls, name: Optional[str] = None) -> Optional[Type]:
        """
        Get a registered parallelization executor class.

        Args:
            name: Name of the parallelization mode. If None, returns None.

        Returns:
            Executor class or None if not found
        """
        if name is None:
            return None

        dotted_path = cls._config["parallelization_registry"].get(name)
        if dotted_path is None:
            return None

        # Import and return the class from dotted path
        try:
            if isinstance(dotted_path, str):
                module_path, class_name = dotted_path.rsplit(".", 1)
                module = import_module(module_path)
                return getattr(module, class_name)
            else:
                # Backward compatibility if already a class
                return dotted_path
        except (ImportError, AttributeError, ValueError) as e:
            warnings.warn(f"Failed to import executor {name} from {dotted_path}: {e}")
            return None

    @classmethod
    def set(cls, key: str, value: Any, local: bool = False) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Configuration value
            local: If True, sets thread-local override; if False, sets global
        """
        if not cls._initialized:
            cls.init()

        if local:
            overrides = cls._context_overrides.get().copy()
            overrides[key] = value
            cls._context_overrides.set(overrides)
        else:
            cls._config[key] = value

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with thread-local override priority.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        if not cls._initialized:
            cls.init()

        # Check context-local overrides first
        overrides = cls._context_overrides.get()
        if key in overrides:
            return overrides[key]

        # Fall back to global config
        return cls._config.get(key, default)

    @classmethod
    @contextmanager
    def use(cls, **settings):
        """
        Context manager for temporary configuration settings.

        Args:
            **settings: Key-value pairs to temporarily set

        Yields:
            None

        Example:
            with RuntimeContext.use(max_workers=4):
                # Code here runs with max_workers=4
                pass
        """
        if not cls._initialized:
            cls.init()

        # Save current context overrides
        old_overrides = cls._context_overrides.get().copy()

        # Apply new settings
        new_overrides = old_overrides.copy()
        new_overrides.update(settings)
        token = cls._context_overrides.set(new_overrides)

        try:
            yield
        finally:
            # Restore old overrides
            cls._context_overrides.reset(token)

    @classmethod
    def get_available_backends(cls) -> Set[str]:
        """Get set of available optical flow backends."""
        if not cls._initialized:
            cls.init()
        return cls._config["available_backends"].copy()

    @classmethod
    def get_available_parallelization(cls) -> Set[str]:
        """Get set of available parallelization modes."""
        if not cls._initialized:
            cls.init()
        return cls._config["available_parallelization"].copy()

    @classmethod
    def get_available_features(cls) -> Set[str]:
        """Get set of available optional features."""
        if not cls._initialized:
            cls.init()
        return cls._config["available_features"].copy()

    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """Check if a specific backend is available."""
        if not cls._initialized:
            cls.init()
        return backend in cls._config["available_backends"]

    @classmethod
    def is_parallelization_available(cls, mode: str) -> bool:
        """Check if a specific parallelization mode is available."""
        if not cls._initialized:
            cls.init()
        return mode in cls._config["available_parallelization"]

    @classmethod
    def is_feature_available(cls, feature: str) -> bool:
        """Check if a specific feature is available."""
        if not cls._initialized:
            cls.init()
        return feature in cls._config["available_features"]

    @classmethod
    def require_backend(cls, backend: str) -> None:
        """
        Require a specific backend to be available.

        Args:
            backend: Backend name to require

        Raises:
            RuntimeError: If backend is not available
        """
        if not cls.is_backend_available(backend):
            available = ", ".join(sorted(cls.get_available_backends()))
            raise RuntimeError(
                f"Backend '{backend}' is not available. "
                f"Available backends: {available}. "
                f"You may need to install additional packages."
            )

    @classmethod
    def require_parallelization(cls, mode: str) -> None:
        """
        Require a specific parallelization mode to be available.

        Args:
            mode: Parallelization mode to require

        Raises:
            RuntimeError: If mode is not available
        """
        if not cls.is_parallelization_available(mode):
            available = ", ".join(sorted(cls.get_available_parallelization()))
            raise RuntimeError(
                f"Parallelization mode '{mode}' is not available. "
                f"Available modes: {available}"
            )

    @classmethod
    def get_optimal_parallelization(cls, backend: Optional[str] = None) -> str:
        """
        Get the optimal parallelization mode for a given backend.

        Args:
            backend: Optional backend name to consider

        Returns:
            Recommended parallelization mode
        """
        if not cls._initialized:
            cls.init()

        # Backend-specific recommendations
        if backend == "raft-2p":
            # RAFT-2P likely uses GPU, so sequential or threading is better
            if cls.is_parallelization_available("threading"):
                return "threading"
            return "sequential"

        # For CPU-based backends, prefer multiprocessing if available
        if backend == "variational":
            if cls.is_parallelization_available("multiprocessing"):
                return "multiprocessing"
            elif cls.is_parallelization_available("threading"):
                return "threading"

        # Default fallback
        if cls.is_parallelization_available("multiprocessing"):
            return "multiprocessing"
        elif cls.is_parallelization_available("threading"):
            return "threading"

        return "sequential"

    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get a summary of all runtime information.

        Returns:
            Dictionary with all runtime context information
        """
        if not cls._initialized:
            cls.init()

        return {
            "backends": sorted(cls.get_available_backends()),
            "parallelization": sorted(cls.get_available_parallelization()),
            "features": sorted(cls.get_available_features()),
            "max_workers": cls.get("max_workers"),
            "platform": sys.platform,
            "python_version": sys.version,
        }

    @classmethod
    def print_info(cls) -> None:
        """Print a formatted summary of runtime information."""
        info = cls.get_info()

        print("PyFlowReg Runtime Information")
        print("=" * 40)
        print(f"Platform: {info['platform']}")
        print(f"Python: {info['python_version'].split()[0]}")
        print(f"Max Workers: {info['max_workers']}")
        print("\nAvailable Backends:")
        for backend in info["backends"]:
            print(f"  - {backend}")
        print("\nAvailable Parallelization:")
        for mode in info["parallelization"]:
            print(f"  - {mode}")
        print("\nAvailable Features:")
        for feature in info["features"]:
            print(f"  - {feature}")

    @classmethod
    def snapshot(cls, workload: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a snapshot of the current runtime context for multiprocessing.

        Args:
            workload: Optional workload identifier for targeted snapshot

        Returns:
            Dictionary containing serializable runtime state
        """
        if not cls._initialized:
            cls.init()

        snapshot = {
            "config": cls._config.copy(),
            "overrides": cls._context_overrides.get().copy(),
            "workload": workload,
        }

        # Ensure registry contains dotted paths, not classes
        snapshot["config"]["parallelization_registry"] = {
            name: (
                path
                if isinstance(path, str)
                else f"{path.__module__}.{path.__qualname__}"
                if hasattr(path, "__module__")
                else str(path)
            )
            for name, path in cls._config["parallelization_registry"].items()
        }

        # Convert sets to lists for JSON serialization
        for key in [
            "available_features",
            "available_parallelization",
            "available_backends",
        ]:
            if key in snapshot["config"] and isinstance(snapshot["config"][key], set):
                snapshot["config"][key] = list(snapshot["config"][key])

        return snapshot

    @classmethod
    def from_env(cls, env_var: str = "PYFLOWREG_CONTEXT") -> None:
        """
        Load runtime context from environment variable (for worker processes).

        Args:
            env_var: Name of environment variable containing JSON snapshot
        """
        snapshot_json = os.environ.get(env_var)
        if snapshot_json:
            try:
                snapshot = json.loads(snapshot_json)
                cls.from_snapshot(snapshot)
            except (json.JSONDecodeError, KeyError) as e:
                warnings.warn(f"Failed to load runtime context from {env_var}: {e}")

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> None:
        """
        Restore runtime context from a snapshot.

        Args:
            snapshot: Dictionary containing runtime state
        """
        if "config" in snapshot:
            config = snapshot["config"]

            # Convert lists back to sets
            for key in [
                "available_features",
                "available_parallelization",
                "available_backends",
            ]:
                if key in config and isinstance(config[key], list):
                    config[key] = set(config[key])

            cls._config.update(config)

        if "overrides" in snapshot:
            cls._context_overrides.set(snapshot["overrides"])

        cls._initialized = True

    @classmethod
    def to_env(
        cls, env_var: str = "PYFLOWREG_CONTEXT", workload: Optional[str] = None
    ) -> None:
        """
        Export runtime context to environment variable for worker processes.

        Args:
            env_var: Name of environment variable to set
            workload: Optional workload identifier
        """
        snapshot = cls.snapshot(workload)
        os.environ[env_var] = json.dumps(snapshot)


# DO NOT auto-initialize on import - let users control when initialization happens
# RuntimeContext.init()
