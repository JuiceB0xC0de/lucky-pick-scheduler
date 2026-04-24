# Contributing

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install pytest
```

## Development Workflow

1. Create a feature branch.
2. Make focused changes with clear commit messages.
3. Add or update tests in `tests/`.
4. Run `pytest -q` locally.
5. Open a PR with context, scope, and validation notes.

## Pull Request Checklist

- [ ] Code compiles/runs locally
- [ ] Tests pass locally
- [ ] README/docs updated if behavior changed
- [ ] Backward compatibility impact considered
