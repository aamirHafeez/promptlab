"""CLI smoke / integration tests via Click's test runner."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from promptlab.cli import main


def test_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "pytest for your prompts" in result.output


def test_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0


def test_init_creates_config(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0
        assert Path("promptlab.yml").exists()


def test_init_refuses_overwrite(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["init"])
        assert result.exit_code != 0


def test_quick_dry_run(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["quick", "hello", "-p", "ollama/llama3", "--dry-run"])
        assert result.exit_code == 0
        assert "passed" in result.output


def test_run_dry_run_end_to_end(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(main, ["init"])
        result = runner.invoke(main, ["run", "--dry-run"])
        assert result.exit_code == 0
        assert Path(".promptlab/last_run.json").exists()


def test_quick_unknown_provider(tmp_path: Path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["quick", "hi", "-p", "bogus/model", "--dry-run"])
        assert result.exit_code != 0
