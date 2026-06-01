# Roadmap

Planned features and improvements for PromptLab. Contributions welcome —
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Evaluators

- **`similarity`** — Cosine similarity to an expected output using an
  embedding model. Requires choosing a default embedding provider and
  managing the additional API cost. Tracked in the evaluator registry as a
  stub today.

## Providers

- **More providers** — Mistral, Cohere, AWS Bedrock, Azure OpenAI
  (dedicated module), and other OpenAI-compatible endpoints.
- **Streaming support** — Stream responses for latency-sensitive
  benchmarks and real-time progress display.

## Reliability

- ~~**Retry with backoff**~~ — ✅ Shipped in v0.1.0. Retries on HTTP 429
  and 5xx with exponential backoff (max 2 retries).

## Reporting

- **Richer HTML report** — Charts for latency/cost comparison, collapsible
  output cells, dark mode support.
- **Markdown export** — Generate a comparison table in Markdown for easy
  pasting into GitHub issues and PRs.
- **CI integration** — GitHub Actions summary annotations and PR comment
  bots for prompt quality gates.

## Developer experience

- **`promptlab watch`** — Re-run tests on config file changes.
- **Plugin system** — Allow third-party evaluators and providers to be
  registered via entry points.
- **Config inheritance** — Share common provider/assertion configs across
  multiple test suites.
