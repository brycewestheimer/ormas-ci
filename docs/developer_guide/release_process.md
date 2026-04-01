# Release Process

## Pre-Release Checklist

1. **All CI checks pass** on the `main` branch (lint, type check,
   tests on Python 3.11/3.12/3.13, docs build)

2. **Version bump** in `pyproject.toml`:
   ```toml
   [project]
   version = "X.Y.Z"
   ```

3. **Update CHANGELOG** with a summary of changes since the last
   release

4. **Documentation build clean:**
   ```bash
   sphinx-build -W -b html docs docs/_build/html
   ```
   The `-W` flag treats warnings as errors -- all cross-references and
   autodoc targets must resolve.

## Building the Package

```bash
python -m build
```

This produces both an sdist (`.tar.gz`) and a wheel (`.whl`) in the
`dist/` directory.

### Smoke Test

```bash
python -m venv /tmp/test-install
/tmp/test-install/bin/pip install dist/*.whl
/tmp/test-install/bin/python -c "from pyscf.ormas_ci import ORMASFCISolver; print('OK')"
```

## Publishing to PyPI

```bash
pip install twine
twine upload dist/*
```

## Documentation Deployment

Documentation is deployed automatically to GitHub Pages when changes
are merged to `main`. The CI pipeline (`.github/workflows/ci.yml`)
builds the Sphinx documentation and deploys via the `deploy-docs` job.

No manual deployment step is needed.

## Post-Release

1. Tag the release: `git tag vX.Y.Z && git push origin vX.Y.Z`
2. Create a GitHub release from the tag with release notes
3. Verify the package installs from PyPI: `pip install pyscf-ormas-ci==X.Y.Z`
4. Verify the documentation is updated at the GitHub Pages URL
