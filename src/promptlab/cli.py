"""PromptLab command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from promptlab import __version__
from promptlab.config import Config, ConfigError, ProviderConfig, load_config
from promptlab.providers import available_providers
from promptlab.reporter import console, export_html, export_json, html_from_dict, print_report
from promptlab.runner import Runner

LAST_RUN = Path(".promptlab/last_run.json")

STARTER_CONFIG = """\
# promptlab.yml — created by `promptlab init`
description: "My first prompt test"

providers:
  - id: anthropic/claude-sonnet-4-6
    temperature: 0.7
  - id: openai/gpt-4o
    temperature: 0.7
  # - id: ollama/llama3        # local, no API key needed

prompts:
  - "You are a helpful assistant. Answer concisely: {{query}}"
  - "You are a friendly expert. Answer concisely: {{query}}"

test_cases:
  - vars:
      query: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
      - type: max_latency
        value: 5000
"""


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="promptlab")
def main() -> None:
    """PromptLab — pytest for your prompts.

    Test, compare, and benchmark prompts across OpenAI, Anthropic, Google,
    and local Ollama models.
    """


def _execute(
    config: Config, dry_run: bool, concurrency: int, judge: str | None, json_path: str | None, html_path: str | None
) -> int:
    from rich.progress import Progress, SpinnerColumn, TextColumn

    runner = Runner(config, dry_run=dry_run, concurrency=concurrency, judge_id=judge)
    total = len(config.providers) * len(config.prompts) * max(1, len(config.test_cases))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        task = progress.add_task(f"Running {total} cases...", total=total)
        report = runner.run(progress_cb=lambda: progress.advance(task))

    print_report(report)

    LAST_RUN.parent.mkdir(parents=True, exist_ok=True)
    export_json(report, LAST_RUN)
    if json_path:
        export_json(report, json_path)
        console.print(f"[dim]JSON written to {json_path}[/dim]")
    if html_path:
        export_html(report, html_path)
        console.print(f"[dim]HTML written to {html_path}[/dim]")

    return 0 if report.failed == 0 else 1


@main.command()
@click.argument("config_path", type=click.Path(exists=True), default="promptlab.yml")
@click.option("--dry-run", is_flag=True, help="Build the matrix but send no API requests.")
@click.option("--concurrency", default=5, show_default=True, help="Max parallel requests.")
@click.option("--judge", default=None, help="Provider id for llm_judge, e.g. anthropic/claude-sonnet-4-6.")
@click.option("--json", "json_path", default=None, help="Also write a JSON report to this path.")
@click.option("--html", "html_path", default=None, help="Also write an HTML report to this path.")
def run(
    config_path: str, dry_run: bool, concurrency: int, judge: str | None, json_path: str | None, html_path: str | None
) -> None:
    """Run a prompt test suite from a YAML config."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    raise SystemExit(_execute(config, dry_run, concurrency, judge, json_path, html_path))


@main.command()
@click.argument("prompt")
@click.option(
    "--provider",
    "-p",
    "providers",
    multiple=True,
    help="Provider id (repeatable). Default: anthropic/claude-sonnet-4-6.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be sent without calling any API.")
@click.option("--concurrency", default=5, show_default=True)
def quick(prompt: str, providers: tuple[str, ...], dry_run: bool, concurrency: int) -> None:
    """Test a single PROMPT across one or more models instantly.

    Example: promptlab quick "Explain recursion" -p openai/gpt-4o -p ollama/llama3
    """
    ids = list(providers) or ["anthropic/claude-sonnet-4-6"]
    for pid in ids:
        if pid.split("/", 1)[0] not in available_providers():
            raise click.ClickException(f"Unknown provider in {pid!r}. Available: {', '.join(available_providers())}.")
    config = Config(
        description=f"quick: {prompt[:60]}",
        providers=[ProviderConfig(id=pid) for pid in ids],
        prompts=[prompt],
        test_cases=[],
    )
    raise SystemExit(_execute(config, dry_run, concurrency, None, None, None))


@main.command()
@click.argument("path", type=click.Path(), default="promptlab.yml")
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def init(path: str, force: bool) -> None:
    """Generate a starter promptlab.yml config."""
    target = Path(path)
    if target.exists() and not force:
        raise click.ClickException(f"{path} already exists. Use --force to overwrite.")
    target.write_text(STARTER_CONFIG, encoding="utf-8")
    console.print(f"[green]Created {path}[/green] — edit it, then run [bold]promptlab run[/bold].")


@main.command()
@click.argument("config_path", type=click.Path(exists=True), default="promptlab.yml")
@click.option("--dry-run", is_flag=True)
@click.option("--concurrency", default=5, show_default=True)
def compare(config_path: str, dry_run: bool, concurrency: int) -> None:
    """Run a suite and show prompt variants side by side.

    This is `run` focused on comparison — most useful when your config lists
    two or more prompts.
    """
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    if len(config.prompts) < 2:
        console.print("[yellow]Tip: add 2+ prompts to your config to get a real comparison.[/yellow]")
    raise SystemExit(_execute(config, dry_run, concurrency, None, None, None))


@main.command()
@click.option("--from", "from_path", default=str(LAST_RUN), show_default=True, help="JSON run file to read.")
@click.option("--html", "html_path", default=None, help="Write an HTML report to this path.")
@click.option("--json", "json_path", default=None, help="Copy the JSON report to this path.")
def report(from_path: str, html_path: str | None, json_path: str | None) -> None:
    """Generate an HTML/JSON report from the last run."""
    src = Path(from_path)
    if not src.exists():
        raise click.ClickException(f"No run found at {from_path}. Run `promptlab run` first.")
    data = json.loads(src.read_text(encoding="utf-8"))
    if not html_path and not json_path:
        html_path = "promptlab-report.html"
    if html_path:
        html_from_dict(data, html_path)
        console.print(f"[green]HTML report written to {html_path}[/green]")
    if json_path:
        Path(json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        console.print(f"[green]JSON report written to {json_path}[/green]")


if __name__ == "__main__":
    main()
