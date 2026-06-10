# Z-Alignment

The `pyflowreg.z_align` module implements a stage-based pipeline for depth-shift (z-shift) estimation and correction, ported from the MATLAB patch-based z-shift workflow. The pipeline compares a recording against a reference volume and runs in three stages: (1) build or load a compensated reference volume, (2) estimate per-frame, per-pixel z-shifts and optionally write z-corrected output, and (3) optionally simulate a z-shift-only recording from the reference volume and the estimated z-shifts. Stage completion is tracked in a `status.json` file under the output root; with `resume=True` (the default), stages whose outputs already exist are skipped on re-runs.

## Configuration

All pipeline parameters are collected in the `ZAlignConfig` Pydantic model. Configurations can be constructed directly in Python or loaded from TOML/YAML files via `ZAlignConfig.from_toml()`, `ZAlignConfig.from_yaml()`, or the extension-dispatching `ZAlignConfig.from_file()`.

```{eval-rst}
.. autoclass:: pyflowreg.z_align.config.ZAlignConfig
   :members:
   :exclude-members: model_config, model_fields, model_computed_fields
```

## Pipeline Functions

The stage runners live in `pyflowreg.z_align.pipeline` and are re-exported from the `pyflowreg.z_align` package.

```{eval-rst}
.. autofunction:: pyflowreg.z_align.pipeline.run_stage1
```

```{eval-rst}
.. autofunction:: pyflowreg.z_align.pipeline.run_recording_prealignment
```

`run_recording_prealignment` is also invoked from within `run_stage2` when `ZAlignConfig.prealign_recording` is enabled, so the stage-2 input recording is prealigned before z-shift estimation.

```{eval-rst}
.. autofunction:: pyflowreg.z_align.pipeline.run_stage2
```

```{eval-rst}
.. autofunction:: pyflowreg.z_align.pipeline.run_stage3
```

```{eval-rst}
.. autofunction:: pyflowreg.z_align.pipeline.run_all_stages
```

## Command-Line Interface

The package installs a `pyflowreg-z-align` console script (entry point `pyflowreg.z_align.cli:main`) with a single `run` subcommand:

```bash
# Run full z-align workflow
pyflowreg-z-align run --config z_align.toml

# Run only z-shift estimation/correction
pyflowreg-z-align run --config z_align.toml --stage 2

# Override stage-1 OFOptions from CLI
pyflowreg-z-align run --config z_align.toml --of-params alpha=8 quality_setting=balanced
```

Arguments of the `run` subcommand:

| Argument | Description |
|----------|-------------|
| `--config`, `-c` (required) | Path to the z-align config file (`.toml`/`.yaml`/`.yml`) |
| `--stage`, `-s` | Run only one stage (`1`, `2`, or `3`); default runs all applicable stages |
| `--of-params KEY=VALUE ...` | Override stage-1 `OFOptions` parameters; ignored with a warning when `--stage` is `2` or `3` |

Override values are parsed as booleans (`true`/`false`), integers, floats, or JSON lists/dicts where possible; everything else is passed through as a string.
