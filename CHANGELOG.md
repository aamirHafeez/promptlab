# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed
- Evaluators `max_tokens`, `max_latency`, `max_cost` no longer crash when
  `value` is missing or non-numeric — they return a clear failure instead.
- `matches_regex` no longer crashes on invalid regex patterns.
- OpenAI provider gives a readable error when the API returns no choices.
- Runner wraps every evaluator call in try/except so a buggy assertion can
  never abort an entire run.
- Google provider sends API key via `x-goog-api-key` header (no longer
  leaks into URLs/logs) and surfaces blocked/safety-filtered responses.
- HTML report `_escape()` now also escapes quotes for defense in depth.
- `llm_judge` now takes the **last** integer from the judge response and
  clamps to 1–10, preventing misparsing.
- README install command fixed — was pointing at unpublished PyPI package;
  now uses `pip install git+https://...` with a note about future PyPI
  release. Broken PyPI badges removed.
- Judge provider HTTP client is now closed after the evaluation phase,
  preventing `ResourceWarning` when using `--judge`.

### Changed
- Provider instances are now cached and reused across cells (one per
  unique config), and each provider shares a single `httpx.AsyncClient`
  for connection pooling.
- Duplicate provider IDs with different options (e.g. same model, different
  temperature) are now correctly distinguished via config index.
- Anthropic default `max_tokens` raised from 1024 → 4096.
- Ollama switched from `/api/generate` to `/api/chat` for consistency with
  the other chat-based providers.
- Transient HTTP errors (429, 5xx) are retried up to 2 times with
  exponential backoff.
- Retry now also covers transient network errors (`ConnectError`,
  `TimeoutException`, etc.) — not just HTTP status codes.
- CI now enforces `ruff format --check` alongside `ruff check`.

### Added
- `ROADMAP.md` — the `similarity` evaluator and README now link to a real
  roadmap file.

## [0.1.0] - 2026-06-02

### Added
- Initial release.
- CLI with `quick`, `init`, `run`, `compare`, and `report` commands.
- Providers: OpenAI (and OpenAI-compatible endpoints), Anthropic, Google
  Gemini, and local Ollama.
- Evaluators: `contains`, `not_contains`, `matches_regex`, `starts_with`,
  `is_json`, `json_schema`, `max_tokens`, `max_latency`, `max_cost`,
  `llm_judge`.
- YAML-driven test suites with `{{var}}` templating.
- Rich terminal report plus JSON and HTML export.
- Concurrent execution with a configurable concurrency limit.

### Known limitations
- `similarity` (embedding-based) assertion is not implemented yet.
- Cost figures are approximate and read from per-provider price tables that
  must be kept in sync with vendors' pricing pages.
