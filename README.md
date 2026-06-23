# fue

![PyPI version](https://img.shields.io/pypi/v/fue.svg)

A Python tool that tracks 14-day weather forecasts from the Open-Meteo API to see how they hold up over time. By comparing early predictions to the actual weather, it calculates the "real" uncertainty of a forecast. It’s built for hikers and outdoor enthusiasts who want to know if that 18°C Saturday is a sure thing or just a hopeful guess.

* [GitHub](https://github.com/sebastian-linden/fue/) | [PyPI](https://pypi.org/project/fue/) | [Documentation](https://sebastian-linden.github.io/fue/)
* Created by [Sebastian Linden](https://audrey.feldroy.com/) | GitHub [@sebastian-linden](https://github.com/sebastian-linden) | PyPI [@sebastian-linden](https://pypi.org/user/sebastian-linden/)
* MIT License

## Features

* TODO

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://sebastian-linden.github.io/fue/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:your_username/fue.git
cd fue

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `fue`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

fue was created in 2026 by Sebastian Linden.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
