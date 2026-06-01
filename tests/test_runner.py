"""Tests for the runner using dry-run mode (no network)."""

from __future__ import annotations

from promptlab.config import Config, ProviderConfig
from promptlab.runner import Runner
from promptlab.utils import render, template_vars


def test_dry_run_matrix_size(config):
    report = Runner(config, dry_run=True).run()
    # 2 providers x 2 prompts x 1 case = 4 cells
    assert report.total == 4
    assert all(c.response is not None for c in report.cells)
    assert all(c.error is None for c in report.cells)
    assert all(c.skipped for c in report.cells)
    assert report.failed == 0


def test_dry_run_no_cost(config):
    report = Runner(config, dry_run=True).run()
    assert report.total_cost == 0.0


def test_render_substitutes_vars():
    assert render("Hi {{name}}", {"name": "Sam"}) == "Hi Sam"


def test_render_leaves_unknown_vars():
    assert render("Hi {{name}}", {}) == "Hi {{name}}"


def test_template_vars_extraction():
    assert template_vars("{{a}} and {{ b }}") == ["a", "b"]


def test_rendered_prompt_uses_case_vars(config):
    report = Runner(config, dry_run=True).run()
    assert any("capital of France" in c.rendered_prompt for c in report.cells)


def test_duplicate_provider_id_different_options():
    """B4: two providers sharing the same id but different options must both be exercised."""
    cfg = Config(
        description="dup-id test",
        providers=[
            ProviderConfig(id="openai/gpt-4o", options={"temperature": 0.0}),
            ProviderConfig(id="openai/gpt-4o", options={"temperature": 1.0}),
        ],
        prompts=["Hello"],
        test_cases=[],
    )
    report = Runner(cfg, dry_run=True).run()
    # 2 providers × 1 prompt × 1 (empty) case = 2 cells
    assert report.total == 2
    # Verify cells reference different provider config indices
    indices = {c.provider_cfg_index for c in report.cells}
    assert indices == {0, 1}
