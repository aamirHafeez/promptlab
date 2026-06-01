"""Rich-powered terminal output plus JSON / HTML export."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

from rich.console import Console
from rich.table import Table

from promptlab.runner import RunReport
from promptlab.utils import fmt_cost, truncate

console = Console()


def print_report(report: RunReport) -> None:
    """Render the results matrix to the terminal."""
    if report.description:
        console.print(f"[bold]{report.description}[/bold]\n")

    table = Table(show_lines=False, header_style="bold")
    table.add_column("Result", justify="center")
    table.add_column("Provider", style="cyan")
    table.add_column("Prompt", justify="right")
    table.add_column("Case")
    table.add_column("Latency", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Output / failure")

    for cell in report.cells:
        if cell.error:
            mark, detail = "[red]ERROR[/red]", f"[red]{truncate(cell.error, 50)}[/red]"
            latency = tokens = cost = "-"
        elif cell.skipped:
            mark = "[dim]DRY[/dim]"
            resp = cell.response
            detail = f"[dim]{truncate(resp.text, 50) if resp else 'dry-run'}[/dim]"
            latency = tokens = cost = "-"
        else:
            ok = cell.passed
            mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            resp = cell.response
            latency = f"{resp.latency_ms:.0f}ms" if resp else "-"
            tokens = str(resp.tokens_used) if resp else "-"
            cost = fmt_cost(resp.cost) if resp else "-"
            if ok:
                detail = truncate(resp.text, 50) if resp else ""
            else:
                fails = "; ".join(e.message for e in cell.evaluations if not e.passed and e.message)
                detail = f"[yellow]{truncate(fails, 50)}[/yellow]"
        case_label = cell.case_name or f"#{cell.case_index + 1}"
        table.add_row(mark, cell.provider_id, f"#{cell.prompt_index + 1}", case_label, latency, tokens, cost, detail)

    console.print(table)
    summary_style = "green" if report.failed == 0 else "red"
    console.print(
        f"\n[{summary_style}]{report.passed}/{report.total} passed[/{summary_style}]"
        f"  ·  total cost {fmt_cost(report.total_cost)}"
    )


def _report_to_dict(report: RunReport) -> dict:
    return {
        "description": report.description,
        "summary": {
            "total": report.total,
            "passed": report.passed,
            "failed": report.failed,
            "total_cost": report.total_cost,
        },
        "cells": [
            {
                "provider_id": c.provider_id,
                "prompt_index": c.prompt_index,
                "case_index": c.case_index,
                "case_name": c.case_name,
                "rendered_prompt": c.rendered_prompt,
                "passed": c.passed,
                "error": c.error,
                "response": (
                    {k: v for k, v in asdict(c.response).items() if k != "raw_response"} if c.response else None
                ),
                "evaluations": [asdict(e) for e in c.evaluations],
            }
            for c in report.cells
        ],
    }


def export_json(report: RunReport, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.write_text(json.dumps(_report_to_dict(report), indent=2), encoding="utf-8")
    return path


def export_html(report: RunReport, path: Union[str, Path]) -> Path:
    return html_from_dict(_report_to_dict(report), path)


def html_from_dict(data: dict, path: Union[str, Path]) -> Path:
    """Build an HTML report from the serialized dict (used by `report`)."""
    path = Path(path)
    summary = data.get("summary", {})
    rows = []
    for c in data.get("cells", []):
        status = "error" if c.get("error") else ("pass" if c.get("passed") else "fail")
        out = c.get("error") or ((c.get("response") or {}).get("text", ""))
        rows.append(
            f"<tr class='{status}'><td>{status.upper()}</td><td>{_escape(c.get('provider_id', ''))}</td>"
            f"<td>#{c.get('prompt_index', 0) + 1}</td>"
            f"<td>{_escape(str(c.get('case_name') or c.get('case_index', 0) + 1))}</td>"
            f"<td><pre>{_escape(out)[:400]}</pre></td></tr>"
        )
    cost = fmt_cost(summary.get("total_cost", 0.0))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PromptLab report</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:.5rem;text-align:left;vertical-align:top}}
tr.pass td:first-child{{color:#0a7d27;font-weight:600}}
tr.fail td:first-child{{color:#c0271a;font-weight:600}}
tr.error td:first-child{{color:#9333ea;font-weight:600}}
pre{{margin:0;white-space:pre-wrap;font-size:.85rem}}
</style></head><body>
<h1>PromptLab report</h1>
<p>{_escape(data.get("description", ""))}</p>
<p><strong>{summary.get("passed", 0)}/{summary.get("total", 0)} passed</strong> · total cost {cost}</p>
<table><thead><tr><th>Result</th><th>Provider</th><th>Prompt</th><th>Case</th><th>Output</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
