# Contributing to GR00T-WholeBodyControl

We welcome contributions from the community! Here's how to get started.

## Reporting Issues

- Search [existing issues](https://github.com/NVlabs/GR00T-WholeBodyControl/issues) first
- Open a new issue with a clear description, error messages, and steps to reproduce
- Include your Python version, OS, GPU, and Isaac Lab version

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Run the pre-flight check: `python check_environment.py`
5. Commit and push to your fork
6. Open a pull request against `main`

### Guidelines

- Keep PRs focused on a single change
- Follow existing code style (no linter is enforced, but be consistent)
- Update documentation if your change affects user-facing behavior
- Add yourself to the PR description if you'd like credit

## Development Setup

See the [Installation Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html)
for setting up the training environment.

### Pre-commit security hook

This repo ships a pre-commit scan that blocks commits containing
credential-shaped content (API keys, tokens, embedded git credentials,
personal email addresses). Install it once per clone:

```bash
bash scripts/git-hooks/install.sh
```

It chains any pre-commit hook you already have. For a deliberate
exception, commit with `SKIP_SECURITY_CHECK=1`.

## Questions

For questions, open a [GitHub Discussion](https://github.com/NVlabs/GR00T-WholeBodyControl/issues)
or contact [gear-wbc@nvidia.com](mailto:gear-wbc@nvidia.com).

## License

By contributing, you agree that your contributions will be licensed under the
[Apache 2.0 License](LICENSE).
