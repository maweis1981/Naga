# Contributing

Thanks for your interest in improving Naga.

## Getting Started

1. Create a Python virtual environment.
2. Install dependencies with `pip install -e .` or the dependency list in [README.md](/Users/mawei/ai_workspaces/ngllm/README.md:26).
3. Use the direct CLI (`python -m naga.cli`) for local engine work and the unified CLI (`python -m naga`) for server-backed workflows.

## Development Notes

- Keep the project Apple Silicon focused unless a change is explicitly about portability.
- Prefer small, reviewable pull requests.
- When changing model loading, serving, RAG, or MCP behavior, include a short validation note in the PR description.
- Do not commit local virtualenvs, model caches, secrets, or generated Python bytecode.

## Validation

At minimum, run the narrowest check that matches your change. Examples:

- `python -m naga --help`
- `python -m naga.cli --help`
- `python -m naga.vlm_cli --help`
- `python -m compileall naga`

## Pull Requests

- Describe what changed and why.
- Mention any model, hardware, or environment assumptions.
- Call out security-sensitive behavior changes explicitly.
