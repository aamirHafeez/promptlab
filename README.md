# PromptLab

**pytest for your prompts.** A Python CLI to test, compare, and benchmark prompts across OpenAI, Anthropic, Google Gemini, and local Ollama models.

[![CI](https://github.com/aamirHafeez/promptlab/actions/workflows/ci.yml/badge.svg)](https://github.com/aamirHafeez/promptlab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

---

## Why PromptLab

Prompt changes are code changes, but most people ship them on vibes. PromptLab lets you write declarative test suites — like pytest fixtures — and run the same prompts across multiple models to compare quality, latency, and cost side by side.

- **Python-native.** Built for the language most AI work happens in.
- **Multi-provider.** OpenAI (and any OpenAI-compatible endpoint — Azure, vLLM, LiteLLM), Anthropic, Google Gemini, and local Ollama.
- **Zero-config start.** `promptlab quick "your prompt"` benchmarks instantly.
- **Readable results.** A color-coded Rich table in your terminal; JSON/HTML export for CI.
- **YAML-driven.** Declarative suites you can version-control and diff.

> **Note on the landscape:** In March 2026, [Promptfoo was acquired by OpenAI](https://openai.com/index/openai-to-acquire-promptfoo/). Promptfoo remains open source and multi-provider, so it's not going away — but some teams prefer a tool that isn't owned by a model vendor. PromptLab is independent and Python-first. Pick whichever fits you.

---

## Install

_PyPI release coming soon — install from source for now._

```bash
pip install git+https://github.com/aamirHafeez/promptlab
```

The CLI is `promptlab`. Set whichever API keys you need as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...
# Ollama needs no key — just a local server at http://localhost:11434
```

## Quick start (30 seconds)

```bash
# Try one prompt across two models — add --dry-run to preview with no API calls
promptlab quick "Explain recursion to a 10-year-old" \
  -p anthropic/claude-sonnet-4-6 -p openai/gpt-4o

# Or scaffold a full test suite
promptlab init
promptlab run
```

## Commands

| Command | What it does |
|---------|--------------|
| `promptlab quick "<prompt>"` | Test one prompt across models instantly |
| `promptlab init` | Generate a starter `promptlab.yml` |
| `promptlab run` | Run a full test suite from YAML |
| `promptlab compare` | Run a suite focused on prompt-vs-prompt |
| `promptlab report` | Regenerate an HTML/JSON report from the last run |

Useful flags on `run`: `--dry-run`, `--concurrency N`, `--judge <provider/model>`, `--json out.json`, `--html out.html`.

## Config format

```yaml
# promptlab.yml
description: "Customer support tone test"

providers:
  - id: anthropic/claude-sonnet-4-6
    temperature: 0.7
  - id: openai/gpt-4o
    temperature: 0.7
  - id: ollama/llama3            # local, no key

prompts:
  - "You are a helpful support agent. {{query}}"
  - "You are a friendly, empathetic support specialist. {{query}}"

test_cases:
  - vars:
      query: "My order hasn't arrived yet"
    assert:
      - type: contains
        value: "sorry"
      - type: max_latency
        value: 3000
      - type: llm_judge          # needs --judge <provider/model>
        value: "Response is empathetic and offers a next step"
        threshold: 7
```

## Built-in assertions

| Type | Checks |
|------|--------|
| `contains` / `not_contains` | Substring is / isn't present |
| `matches_regex` | Output matches a regex |
| `starts_with` | Output starts with a prefix |
| `is_json` | Output is (or isn't) valid JSON |
| `json_schema` | Output validates against a JSON Schema |
| `max_tokens` | Completion within a token budget |
| `max_latency` | Response within a latency budget (ms) |
| `max_cost` | Response within a cost budget (USD) |
| `llm_judge` | A judge model scores the output 1–10 against a rubric |
| `similarity` | *(planned — see [ROADMAP](ROADMAP.md))* cosine similarity to expected output |

## Provider / model ids

Use `provider/model`, e.g. `openai/gpt-4o`, `anthropic/claude-sonnet-4-6`, `google/gemini-2.5-pro`, `ollama/llama3`.

Model availability and pricing change frequently. PromptLab keeps approximate prices in each provider module (`src/promptlab/providers/*.py`) — edit them to match the current pricing pages. Cost numbers are estimates, not billing.

## Use in CI

`promptlab run` exits non-zero if any case fails, so you can gate merges on prompt quality:

```yaml
- run: promptlab run --json results.json
```

## Development

```bash
git clone https://github.com/aamirHafeez/promptlab
cd promptlab
pip install -e ".[dev]"
pytest          # tests (all run offline via --dry-run / fixtures)
ruff check src  # lint
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including embedding-based
similarity, more providers, richer HTML reports, and a plugin system.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup
instructions and guidelines. All changes are tracked in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
