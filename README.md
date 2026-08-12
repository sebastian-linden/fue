# pyfue (Forecast Uncertainty Estimation)

[![PyPI version](https://img.shields.io/pypi/v/pyfue.svg)](https://pypi.org/project/pyfue/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg)](https://sebastian-linden.github.io/pyfue/)
[![Code License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Abstract

Welcome to the `pyfue` package. This package enables you to estimate empiric error bounds for weather forecasts of your city. Numerical weather simulations used for todays forecasts will produce a timeseries of numbers (e.g. temperature, precipitation sum) that are in reality just the the most likely values of an underlying probability distribution with some variance around the mean. Since the error bounds of these distributions are generally not available, I developed this package to estimate these error bounds empirically. This package enables you to train linear or non-linear models on historic forecasts errors to predict expected error bounds for forecasts of your city.


## Background

Being an outdoor enthusiast, I am looking at the weather forecasts regularly and more than once I had to change my plans on short notice, because the weather turned out to be less inviting than I had anticipated. Naturally I came ask myself how reliable weather forecasts really are, especially in Aachen, where the weather isn't very stable. I initially planned to build my own Arduino weather station on the balcony and then compare forecasts from previous days to what my weather station would measure.

I postponed this project for a while and only thought about this idea again when taking a graduate course called "Sustainable Computational Engineering" by Dr. sc. Anil Yildiz at RWTH Aachen University. As a final project we would have to write our own Python package that follows state of the art conventions for scientific programming. I took this opportunity to turn a hobby project idea into my first Python package.

## The use of pen & paper, human and artificial intelligence in this project

Since AI generated content has flooded the internet, I came to value more and more honest human written text. On the other side, I value well written code and easy to understand documentation. Therefore, I want to state very clearly how and where I used AI in this project.

I initialized this project using the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template, which may or may not contain AI generated markdown files. From that starting point, I used:
- **Pen & Paper** to come up with the basic math for my analysis and to draft and revise my package architecture
- **Artificial Intelligence** to implement specific features (one at a time), fix bugs, write tests, implement logging and write all of the inline documentation (i.e. docstrings) as well as some parts of the long-form documentation.
- **My Human Intelligence** to plan the package development roadmap, proof-read, correct and accept AI generated code and documentation, write most long-form documentation.

Most of the following sections of this README file are also AI generated and checked by me.

## Features

- **Multi-Model Architecture**: Seamlessly swap between standard Linear Regression models and non-linear, ensemble-based Multi-Layer Perceptrons (MLPs).
- **Scientifically Rigorous Pipeline**: Avoids geographical and temporal data leakage using a custom chronological multi-city data splitting engine.
- **Automated Data Ingestion**: Built-in client for the Open-Meteo API to fetch, parse, and pair historical forecasts with ground-truth measurements.
- **Professional CLI**: A Typer-powered Command Line Interface for automated data downloading, dataset summarization, and headless model tuning.
- **Physical Constraints**: Implements target log-transformations and intelligent feature engineering to ensure uncertainty boundaries never violate physical reality.

## Installation

**End-User Installation (PyPI)**
The easiest way to install `pyfue` is via pip:
```bash
pip install pyfue
```

**Developer Installation**
If you want to build pyfue from source or contribute to the project:

```bash
# 1. Clone the repository
git clone [https://github.com/YourUsername/pyfue.git](https://github.com/YourUsername/pyfue.git)
cd pyfue

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: call venv\Scripts\activate.bat

# 3. Install locally in editable mode
pip install -e .
```

**Quick Setup**
To generate the basic folder structure for the package to run without issues:

```bash
# This will guide you through the process
pyfue init
```

## Project Structure
```txt
pyfue/
├── src/pyfue/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract base class
│   │   ├── linear_model.py         # Linear regression subclass
│   │   ├── ml_model.py             # MLP subclass
│   │   └── preprocessor.py         # Stateful data transformations
│   ├── __init__.py
│   ├── cli.py                      # Typer CLI application
│   ├── config.py                   # Global configuration & constants
│   ├── data.py                     # Dataset generation and training/validation splitting 
│   ├── forecast.py                 # Inference and visualization module
│   ├── openmeteoclient.py          # API interaction
│   └── utils.py                    # Math & geo helpers (e.g., Haversine)
├── tests/                          # Pytest suite
├── docs/                           # Sphinx documentation
└── README.md
```

## Documentation
Comprehensive documentation including API references, mathematical methodologies, and workflow tutorials are hosted on [GitHub Pages](https://sebastian-linden.github.io/pyfue/index.html).

## Data
The data, that I collected while working on this project can be found on the Coscine data repository under the following link:

```
https://0d3c12b3-5332-4892-80b6-584d9141ac31.global.datastorage.nrw/forecasts.csv?X-Amz-Expires=86400&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AAB10CA81B3CA758256A%2F20260812%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260812T173713Z&X-Amz-SignedHeaders=host&X-Amz-Signature=b10e1c9d7037309bf4a4382ca2eb6abdd0bc7e9374d85ca8197d599f12907040
```

## Author

pyfue was created in 2026 by Sebastian Linden and co-authored by Google's Gemini AI.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.


## License & Attribution

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Data Credit
This package integrates weather data provided by [Open-Meteo.com](https://open-meteo.com/). The API data is offered under the **Attribution 4.0 International (CC BY 4.0)** license.

Weather data by <a href="https://open-meteo.com/">Open-Meteo.com</a>