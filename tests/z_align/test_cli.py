"""
Tests for z-align CLI.
"""

from __future__ import annotations

import argparse

import pytest

from pyflowreg.z_align.config import ZAlignConfig
from pyflowreg.z_align import cli


class TestCLIParsing:
    """Test helper parsing functions."""

    def test_parse_value_scalars(self):
        assert cli._parse_value("true") is True
        assert cli._parse_value("false") is False
        assert cli._parse_value("12") == 12
        assert cli._parse_value("1.25") == 1.25
        assert cli._parse_value("quality") == "quality"

    def test_parse_value_json(self):
        assert cli._parse_value("[1, 2, 3]") == [1, 2, 3]
        assert cli._parse_value('{"a": 1}') == {"a": 1}

    def test_parse_overrides(self):
        out = cli._parse_overrides(
            ["alpha=5", "write_simulated=false", "quality_setting=balanced"]
        )
        assert out["alpha"] == 5
        assert out["write_simulated"] is False
        assert out["quality_setting"] == "balanced"


class TestCLIRouting:
    """Test subcommand routing behavior."""

    @pytest.fixture
    def config_file(self, tmp_path):
        cfg_file = tmp_path / "z_align.toml"
        cfg_file.write_text(
            "\n".join(
                [
                    f'root = "{tmp_path.as_posix()}"',
                    'input_file = "compensated.tiff"',
                ]
            ),
            encoding="utf-8",
        )
        return cfg_file

    def test_cmd_run_stage1(self, config_file, monkeypatch):
        cfg = ZAlignConfig(root=config_file.parent, input_file="compensated.tiff")
        called = {"stage1": 0}

        monkeypatch.setattr(cli.ZAlignConfig, "from_file", lambda _p: cfg)
        monkeypatch.setattr(
            cli,
            "run_stage1",
            lambda config, overrides=None: called.__setitem__(
                "stage1", called["stage1"] + 1
            ),
        )
        monkeypatch.setattr(
            cli, "run_stage2", lambda *args, **kwargs: pytest.fail("unexpected stage2")
        )
        monkeypatch.setattr(
            cli, "run_stage3", lambda *args, **kwargs: pytest.fail("unexpected stage3")
        )
        monkeypatch.setattr(
            cli,
            "run_all_stages",
            lambda *args, **kwargs: pytest.fail("unexpected run_all_stages"),
        )

        args = argparse.Namespace(
            config=str(config_file),
            stage="1",
            of_params=["alpha=8", "quality_setting=balanced"],
        )
        cli.cmd_run(args)
        assert called["stage1"] == 1

    def test_cmd_run_stage2(self, config_file, monkeypatch):
        cfg = ZAlignConfig(root=config_file.parent, input_file="compensated.tiff")
        called = {"stage2": 0}

        monkeypatch.setattr(cli.ZAlignConfig, "from_file", lambda _p: cfg)
        monkeypatch.setattr(
            cli,
            "run_stage2",
            lambda config: called.__setitem__("stage2", called["stage2"] + 1),
        )
        monkeypatch.setattr(
            cli, "run_stage1", lambda *args, **kwargs: pytest.fail("unexpected stage1")
        )

        args = argparse.Namespace(config=str(config_file), stage="2", of_params=None)
        cli.cmd_run(args)
        assert called["stage2"] == 1

    @pytest.mark.parametrize("stage", ["2", "3"])
    def test_cmd_run_warns_on_ignored_of_params(
        self, config_file, monkeypatch, capsys, stage
    ):
        cfg = ZAlignConfig(root=config_file.parent, input_file="compensated.tiff")

        monkeypatch.setattr(cli.ZAlignConfig, "from_file", lambda _p: cfg)
        monkeypatch.setattr(cli, "run_stage2", lambda config: None)
        monkeypatch.setattr(cli, "run_stage3", lambda config: None)

        args = argparse.Namespace(
            config=str(config_file), stage=stage, of_params=["alpha=8"]
        )
        cli.cmd_run(args)

        out = capsys.readouterr().out
        assert "--of-params" in out
        assert "ignored" in out

    def test_cmd_run_all_stages(self, config_file, monkeypatch):
        cfg = ZAlignConfig(root=config_file.parent, input_file="compensated.tiff")
        called = {"all": 0}

        monkeypatch.setattr(cli.ZAlignConfig, "from_file", lambda _p: cfg)
        monkeypatch.setattr(
            cli,
            "run_all_stages",
            lambda config, overrides=None: called.__setitem__("all", called["all"] + 1),
        )
        monkeypatch.setattr(
            cli, "run_stage1", lambda *args, **kwargs: pytest.fail("unexpected stage1")
        )
        monkeypatch.setattr(
            cli, "run_stage2", lambda *args, **kwargs: pytest.fail("unexpected stage2")
        )
        monkeypatch.setattr(
            cli, "run_stage3", lambda *args, **kwargs: pytest.fail("unexpected stage3")
        )

        args = argparse.Namespace(
            config=str(config_file),
            stage=None,
            of_params=["alpha=5"],
        )
        cli.cmd_run(args)
        assert called["all"] == 1

    def test_cmd_run_missing_config_exits(self, tmp_path):
        args = argparse.Namespace(
            config=str(tmp_path / "missing.toml"),
            stage=None,
            of_params=None,
        )
        with pytest.raises(SystemExit, match="1"):
            cli.cmd_run(args)

    def test_main_without_subcommand_exits(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["pyflowreg-z-align"])
        with pytest.raises(SystemExit, match="1"):
            cli.main()
