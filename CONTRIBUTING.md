# Contributing to PromptLab

Thanks for your interest! Contributions of all sizes are welcome.

## Getting set up

```bash
git clone https://github.com/aamirHafeez/promptlab
cd promptlab
pip install -e ".[dev]"
```

## Before you open a PR

Run the checks locally — CI runs the same ones:

```bash
ruff check src tests      # lint
ruff format --check src tests
mypy src                  # type check
pytest                    # tests (all offline)
```

All tests run without network access or API keys (they use `--dry-run` and
fixtures), so the suite is fast and deterministic. Please keep it that way:
mock or dry-run anything that would hit a real provider.

## Adding a provider

1. Create `src/promptlab/providers/<name>.py` with a class subclassing
   `LLMProvider`, implementing async `complete(...)` and returning a
   `ProviderResponse`.
2. Register it in `src/promptlab/providers/__init__.py`.
3. Add a price table if the provider is paid (clearly marked as approximate).
4. Add a test that exercises it in dry-run / mocked form.

## Adding an evaluator

Add a function to `src/promptlab/evaluators.py` decorated with
`@register("your_type")`. It receives `(response, assertion, context)` and
returns an `EvalResult`. Add tests in `tests/test_evaluators.py`.

## Reporting bugs

Open an issue with a minimal `promptlab.yml` (with secrets removed) and the
command you ran. `--dry-run` output is often enough to reproduce.

## Code of conduct

Be kind. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).
