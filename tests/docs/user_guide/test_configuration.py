"""
Executable-documentation tests for docs/user_guide/configuration.md.

Each test runs the extracted snippet from
``docs/snippets/user_guide/configuration/`` via the ``snippet_runner``
fixture and asserts on the resulting namespace. Targeted companion tests
verify the prose claims of the page (defaults, validation behavior,
preset-to-level mapping) directly against ``OFOptions``.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from pyflowreg.motion_correction import OFOptions
from pyflowreg.motion_correction.OF_options import (
    ChannelNormalization,
    OutputFormat,
    QualitySetting,
)

pytestmark = pytest.mark.docs_example


class TestConfigurationQualitySettings:
    """docs/user_guide/configuration.md, section "Quality Settings"."""

    def test_quality_settings_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/quality_settings.py")

        # The snippet constructs fast, balanced, then quality; the final
        # binding is the "quality" preset (the documented default).
        options = ns["options"]
        assert isinstance(options, OFOptions)
        assert options.quality_setting == QualitySetting.QUALITY
        assert options.effective_min_level == 0
        assert OFOptions().quality_setting == QualitySetting.QUALITY

    @pytest.mark.parametrize(
        "quality_setting,expected_min_level",
        [("fast", 6), ("balanced", 4), ("quality", 0)],
    )
    def test_quality_settings_effective_min_level(
        self, quality_setting, expected_min_level
    ):
        """Page prose: fast stops at level 6, balanced at 4, quality at 0."""
        options = OFOptions(quality_setting=quality_setting)
        assert options.effective_min_level == expected_min_level

    def test_quality_settings_min_level_override_switches_to_custom(self):
        """Page prose: a non-negative min_level switches the preset to custom."""
        options = OFOptions(min_level=2)
        assert options.quality_setting == QualitySetting.CUSTOM
        assert options.effective_min_level == 2


class TestConfigurationCoreParameters:
    """docs/user_guide/configuration.md, section "Core Optical Flow Parameters"."""

    def test_core_parameters_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/core_parameters.py")

        options = ns["options"]
        assert isinstance(options, OFOptions)
        assert options.alpha == (1.5, 1.5)
        assert options.iterations == 50
        assert options.levels == 100
        assert options.eta == 0.8
        assert options.min_level == -1
        assert options.a_smooth == 1.0
        assert options.a_data == 0.45
        assert options.update_lag == 5

        # Every value shown in the snippet is documented as the default.
        defaults = OFOptions()
        for field in (
            "alpha",
            "iterations",
            "levels",
            "eta",
            "min_level",
            "a_smooth",
            "a_data",
            "update_lag",
        ):
            assert getattr(options, field) == getattr(defaults, field)

    def test_core_parameters_scalar_alpha_expands(self):
        """Page prose: a scalar alpha is expanded to (alpha, alpha)."""
        assert OFOptions(alpha=2.0).alpha == (2.0, 2.0)


class TestConfigurationTemporalBinning:
    """docs/user_guide/configuration.md, section "Temporal Binning"."""

    def test_temporal_binning_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/temporal_binning.py")

        options = ns["options"]
        assert options.bin_size == 2
        # Page prose: default 1 (no binning).
        assert OFOptions().bin_size == 1


class TestConfigurationGaussianFiltering:
    """docs/user_guide/configuration.md, section "Gaussian Filtering"."""

    def test_gaussian_filtering_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/gaussian_filtering.py")

        # Final binding is the per-channel form for 2 channels.
        options = ns["options"]
        assert options.sigma == [[1.0, 1.0, 0.1], [2.0, 2.0, 0.2]]

    def test_gaussian_filtering_single_triple_normalizes(self):
        """A single [sx, sy, st] triple is stored as one (1, 3) row."""
        options = OFOptions(sigma=[1.0, 1.0, 0.1])
        assert options.sigma == [[1.0, 1.0, 0.1]]

    def test_gaussian_filtering_default(self):
        """Page prose: the default is [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]."""
        assert OFOptions().sigma == [[1.0, 1.0, 0.1], [1.0, 1.0, 0.1]]


class TestConfigurationChannelNormalization:
    """docs/user_guide/configuration.md, section "Channel Normalization"."""

    def test_channel_normalization_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/channel_normalization.py")

        # Final binding is the "separate" mode.
        options = ns["options"]
        assert options.channel_normalization == ChannelNormalization.SEPARATE
        # Page prose: "joint" is the default.
        assert OFOptions().channel_normalization == ChannelNormalization.JOINT


class TestConfigurationFlowBackend:
    """docs/user_guide/configuration.md, section "Available Flow Backends"."""

    def test_flow_backend_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/flow_backend.py")

        options = ns["options"]
        assert options.flow_backend == "flowreg"
        assert options.backend_params == {}
        # Page prose: "flowreg" is the default backend.
        defaults = OFOptions()
        assert defaults.flow_backend == "flowreg"
        assert defaults.backend_params == {}


class TestConfigurationReferenceFrames:
    """docs/user_guide/configuration.md, section "Fixed Reference Frames"."""

    def test_reference_frames_executes(self, materialize_input, snippet_runner):
        materialize_input("raw_video.h5", shape=(24, 32, 48, 2))
        ns = snippet_runner("user_guide/configuration/reference_frames.py")

        # The precomputed reference is the mean of frames 10:21 -> (H, W, C).
        reference_array = ns["reference_array"]
        assert reference_array.shape == (32, 48, 2)

        # Final binding stores the ndarray reference directly.
        options = ns["options"]
        assert isinstance(options.reference_frames, np.ndarray)
        np.testing.assert_array_equal(options.reference_frames, reference_array)

    def test_reference_frames_default(self):
        """Page prose: the default is list(range(50, 500))."""
        assert OFOptions().reference_frames == list(range(50, 500))

    @pytest.mark.parametrize(
        "reference_frames",
        [[0], list(range(100, 200)), "reference.tif"],
    )
    def test_reference_frames_accepted_forms(self, reference_frames):
        """Indices and file paths are stored unchanged on the options."""
        options = OFOptions(reference_frames=reference_frames)
        assert options.reference_frames == reference_frames


class TestConfigurationUpdateReference:
    """docs/user_guide/configuration.md, section "Updating the Reference Frame"."""

    def test_update_reference_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/update_reference.py")

        options = ns["options"]
        assert options.update_reference is True

        # Page prose: related defaults.
        defaults = OFOptions()
        assert defaults.update_reference is False
        assert defaults.update_initialization_w is True
        assert defaults.n_references == 1


class TestConfigurationCcInitialization:
    """docs/user_guide/configuration.md, section "Cross-Correlation Pre-Alignment"."""

    def test_cc_initialization_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/cc_initialization.py")

        options = ns["options"]
        assert options.cc_initialization is True
        assert options.cc_hw == 256
        assert options.cc_up == 1

        # Page prose: cc_initialization defaults to False, cc_hw to 256,
        # cc_up to 1.
        defaults = OFOptions()
        assert defaults.cc_initialization is False
        assert defaults.cc_hw == 256
        assert defaults.cc_up == 1


class TestConfigurationIoPaths:
    """docs/user_guide/configuration.md, section "Input/Output Paths"."""

    def test_io_paths_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/io_paths.py")

        options = ns["options"]
        assert str(options.input_file) == "raw_video.h5"
        assert options.output_path == Path("results")
        assert options.output_format == OutputFormat.HDF5

        # Page prose: defaults are output_path="results" and format MAT.
        defaults = OFOptions()
        assert defaults.output_path == Path("results")
        assert defaults.output_format == OutputFormat.MAT


class TestConfigurationSaveOutputs:
    """docs/user_guide/configuration.md, section "Saving Displacement Fields"."""

    def test_save_outputs_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/save_outputs.py")

        options = ns["options"]
        assert options.save_w is True
        assert options.output_typename == "single"

        # Page prose: defaults are save_w=False and output_typename="double".
        defaults = OFOptions()
        assert defaults.save_w is False
        assert defaults.output_typename == "double"


class TestConfigurationBufferSize:
    """docs/user_guide/configuration.md, section "Buffer Size"."""

    def test_buffer_size_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/buffer_size.py")

        options = ns["options"]
        assert options.buffer_size == 400
        # Page prose: 400 is the default.
        assert OFOptions().buffer_size == 400


class TestConfigurationValidation:
    """docs/user_guide/configuration.md, section "Validation"."""

    def test_validation_executes(self, snippet_runner):
        ns = snippet_runner("user_guide/configuration/validation.py")

        # The snippet's surviving binding is the expanded scalar alpha.
        options = ns["options"]
        assert options.alpha == (2.0, 2.0)

    def test_validation_negative_alpha_rejected(self):
        """Page prose: invalid values raise a ValidationError."""
        with pytest.raises(ValidationError, match="Alpha must be positive"):
            OFOptions(alpha=-1)

    def test_validation_unknown_field_rejected(self):
        """Page prose: unknown field names are rejected (extra="forbid")."""
        with pytest.raises(ValidationError):
            OFOptions(alhpa=2.0)


class TestConfigurationSaveLoad:
    """docs/user_guide/configuration.md, section "Saving and Loading Configuration"."""

    def test_save_load_executes(self, tmp_path, snippet_runner):
        ns = snippet_runner("user_guide/configuration/save_load.py")

        # The snippet runs in tmp_path, so config.json is written there.
        config_path = tmp_path / "config.json"
        assert config_path.exists()

        # The final binding is the round-tripped options object.
        options = ns["options"]
        assert isinstance(options, OFOptions)
        assert options.quality_setting == QualitySetting.BALANCED
        assert options.alpha == (2.0, 2.0)
        # Untouched fields survive the round trip with their defaults.
        assert options.eta == 0.8
        assert options.output_format == OutputFormat.MAT
