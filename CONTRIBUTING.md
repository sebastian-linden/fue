# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/sebastian-linden/fue/issues.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

### Write Documentation

fue could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

To preview the docs locally:

```sh
just docs-serve
```

This starts a local server at http://localhost:8000 with live reload. Edit files in `docs/` or add docstrings to your code (the API reference page is auto-generated).

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/sebastian-linden/fue/issues.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Get Started!

Ready to contribute? Follow these sections in order to set up fue for local development.

## Environment Setup

### Why a Virtual Environment?

A virtual environment isolates your project dependencies from your system Python and other projects. This prevents dependency conflicts and ensures your development environment is reproducible and independent of your machine's global packages. Every contributor working from the same configuration reduces the "works on my machine" problem.

### Prerequisites

Before you begin, ensure you have:

- **Python 3.12 or newer**: Check your version with `python3 --version`. If you need to install or upgrade Python, visit [python.org](https://www.python.org/downloads/).
- **uv package manager**: This project uses `uv` for fast, reliable dependency management. Install it by following [uv's installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- **Git**: For version control. Install from [git-scm.com](https://git-scm.com/) if needed.

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub and clone your fork locally:

   ```sh
   git clone git@github.com:your-username/fue.git
   cd fue
   ```

   *Why fork?* Forking gives you your own copy to work on without affecting the main repository. You'll later create pull requests to propose changes back to the original project.

2. **Synchronize dependencies** using uv:

   ```sh
   uv sync
   ```

   *What this does*: `uv sync` creates a virtual environment (stored in `.venv/` by default) and installs all dependencies specified in `pyproject.toml`, including development tools. This single command ensures your environment matches everyone else's.

3. **Verify the environment** by checking that key tools are accessible:

   ```sh
   uv run pytest --version
   uv run ruff --version
   uv run ty --version
   ```

   *Why verify?* This confirms that all development tools are installed and accessible through your virtual environment. You should see version numbers for pytest, ruff (code formatter/linter), and ty (type checker).

### Activating Your Virtual Environment

For most development workflows, you don't need to manually activate the environment. Commands run through `uv run` automatically use the virtual environment.

However, if you prefer to activate it manually (for interactive Python shells or repeated commands):

```sh
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

To deactivate later, run:

```sh
deactivate
```

### Keeping Your Environment Clean

Periodically refresh your environment to catch any issues early:

```sh
uv sync --refresh
```

*Why refresh?* This re-downloads and re-installs all dependencies, ensuring stale or corrupted packages don't cause mysterious failures in testing or type checking.

Before submitting a pull request, run this command to verify your code works with fresh dependencies.

---

## Dependency Management

### Understanding Our Dependency Structure

This project uses `pyproject.toml` to manage dependencies with a clear separation of concerns:

- **Core dependencies** (in the `[project]` section): Required to use fue. Currently includes `typer` (for CLI) and `rich` (for terminal formatting).
- **Development dependency groups** (in `[dependency-groups]`): Only needed when developing or contributing. Organized by purpose:
  - `lint`: Code formatting and linting (`ruff`)
  - `test`: Testing and coverage (`pytest`, `coverage`)
  - `typecheck`: Static type checking (`ty`)
  - `docs`: Documentation generation (`zensical`, `mkdocstrings-python`)

When you run `uv sync`, all development groups are included.

### Adding a New Dependency

Follow this workflow when you need a new library:

#### 1. Determine the Dependency Type

Ask yourself:

- **Is this needed to run the application?** → Add to core `dependencies` in `[project]`
- **Is this only for development, testing, or documentation?** → Add to the appropriate `[dependency-groups]`

For example:
- Adding `httpx` for HTTP requests in the main code? → Core dependency.
- Adding `pytest-mock` to improve test fixtures? → `test` group.
- Adding `black` for code formatting? → `lint` group.

#### 2. Add the Dependency to pyproject.toml

Open `pyproject.toml` and edit the relevant section. For a **core dependency**:

```toml
[project]
dependencies = [
  "typer",
  "rich",
  "your-new-package",  # Add here
]
```

For a **development dependency**, add to the appropriate group. For example, to add a testing dependency:

```toml
[dependency-groups]
test = [
    "coverage",
    "pytest",
    "pytest-mock",  # Add here
]
```

*Why separate groups?* Users who install fue from PyPI only get core dependencies. Developers who clone the repository get everything. This keeps the installed package lightweight for end users.

#### 3. Sync Your Environment

After editing `pyproject.toml`, immediately sync to install the new dependency:

```sh
uv sync
```

#### 4. Verify Installation

Check that the package is importable and works:

```sh
uv run python -c "import your_new_package; print(your_new_package.__version__)"
```

#### 5. Test and Commit

- Run your feature or test that uses the new dependency.
- Verify existing tests still pass: `just qa`
- Commit both the new code and the updated `pyproject.toml`:

  ```sh
  git add src/fue/your_feature.py pyproject.toml
  git commit -m "Add your_new_package for feature X

  - Added your_new_package to [dependency-groups.test]
  - Used in: test_feature_x.py for improved mocking
  "
  ```

### Dependency Constraints and Philosophy

- **Python version**: We support Python 3.12+. Never add dependencies requiring older versions. Check [caniusepython3.com](https://caniusepython3.com/) if unsure about version compatibility.
- **Minimize dependencies**: Each dependency is a maintenance burden. Evaluate whether the package is actively maintained and worth the added complexity.
- **Prefer well-established packages**: Look for packages with many stars on GitHub, active issue responses, and clear documentation.

### Keeping Dependencies Up to Date

Periodically check for security updates and new versions:

```sh
uv pip list --outdated
```

To update a specific dependency:

```sh
uv pip install --upgrade your-package-name
```

To update the lock file and `pyproject.toml`:

```sh
uv sync --upgrade
```

*Why update regularly?* Security patches and bug fixes improve code quality and protect users from vulnerabilities. However, always test thoroughly after updates to catch breaking changes.

---

## Testing & QA

### Why testing matters

Tests verify that changes behave as expected and prevent regressions. Automated quality checks also keep the project maintainable and help contributors understand the expected code style and behavior.

### Running tests with pytest

This repository uses `pytest` with tests located in the `tests/` directory by default. Run the suite directly with:

```sh
uv run pytest
```

To run a specific test file or subset:

```sh
uv run pytest tests/test_fue.py
uv run pytest tests/ -k "forecast"
```

Use this direct command when you are focusing on a particular feature or debugging a single failure.

### Using `just` for automation

The `justfile` provides reusable commands for common development tasks. It wraps `uv` and the project’s preferred tools so you do not need to remember the exact command line.

- `just test` runs the default test suite with `uv run --python=3.14 pytest`.
- `just qa` runs the full quality workflow:
  - `uv run --python=3.14 ruff format .`
  - `uv run --python=3.14 ruff check . --fix`
  - `uv run --python=3.14 ruff check --select I --fix .`
  - `uv run --python=3.14 ty check --output-format=concise .`
  - `uv run --python=3.14 pytest`

Because `just qa` uses the same command structure as the project, it is the recommended check before committing or opening a pull request.

### Coverage and test reporting

If you need coverage information, use the `coverage` target from the `justfile`:

```sh
just coverage
```

This command runs the tests across supported Python versions and generates both a terminal report and an HTML report.

### Debugging test failures

When a test fails and you want to investigate interactively, use:

```sh
just pdb -- <pytest-args>
```

This runs `pytest` with `--pdb` and preserves the project’s standard test invocation pattern.

### Best practices for tests

- Add new tests for every bug fix or feature change.
- Keep tests small and focused on one behavior.
- Use descriptive names for test functions and test files.
- Run `just qa` before committing to ensure formatting, linting, type checking, and tests all pass.

---

## Documentation

### Why documentation matters

Clear documentation makes the project accessible to users and contributors. It explains how to use the code, contributes to FAIR principles by improving findability and reusability, and helps maintain code quality through type hints and docstrings.

### Maintaining the docs/ folder

The `docs/` folder contains Markdown files for the project's documentation, built using `zensical` and `mkdocstrings-python`. To contribute to documentation:

1. **Edit documentation files**:

   - Modify existing files in `docs/` (e.g., `docs/usage.md` for user guides).
   - Add new Markdown files as needed, following the existing structure.

2. **Preview changes locally**:

   ```sh
   just docs-serve
   ```

   This starts a local server at http://localhost:8000 with live reload. Edit files in `docs/` or add docstrings to your code—the API reference page is auto-generated from docstrings.

   *Why preview locally?* Live reload lets you see changes instantly without rebuilding, speeding up documentation writing. If the server doesn't start, ensure port 8000 is free (run `lsof -ti :8000 | xargs kill` if needed).

3. **Build documentation for production**:

   ```sh
   just docs-build
   ```

   This generates the final documentation site in strict mode, failing on warnings to ensure quality.

*Why preview and build?* Local preview lets you see changes immediately, while building ensures the documentation is error-free before publishing.

### Understanding `just` workflows

This project uses `just` as a command runner to simplify common development tasks. `just` reads the `justfile` in the project root and provides shortcuts for complex commands. Instead of remembering long `uv run` invocations, use `just` commands for consistency and ease.

- **List all available commands**: Run `just` or `just --list` to see what's available.
- **Run specific tasks**: For example, `just qa` runs the full quality check suite, which is equivalent to multiple `uv run` commands chained together.
- **Pass arguments**: Some commands like `just test` accept arguments (e.g., `just test tests/test_specific.py`).

*Why use `just`?* It standardizes workflows across contributors, reduces typing, and ensures everyone uses the same tool versions and flags. For documentation, `just docs-serve` is the primary way to preview changes interactively.

### Requirements for type hints and docstrings

This project uses static type checking with `ty` and requires type hints and docstrings for maintainability and clarity.

#### Type Hints

- **Add type annotations** to all function parameters, return values, and variables where possible.
- Use Python's `typing` module for complex types (e.g., `List[str]`, `Optional[int]`).
- Run `just type-check` or `just qa` to verify types—fixes are often suggested automatically.

*Why type hints?* They catch errors early, improve IDE support, and make the code self-documenting for future contributors.

#### Docstrings

- **Write docstrings** for all public functions, classes, and modules using Google-style format.
- Include a brief description, parameters, return values, and examples where helpful.
- Example:

  ```python
  def calculate_forecast_error(actual: float, predicted: float) -> float:
      """Calculate the absolute error between actual and predicted forecast values.

      Args:
          actual: The actual weather value.
          predicted: The predicted weather value.

      Returns:
          The absolute difference between the values.

      Example:
          >>> calculate_forecast_error(20.0, 18.5)
          1.5
      """
      return abs(actual - predicted)
  ```

- Docstrings are auto-generated into the API reference in `docs/api.md`.

*Why docstrings?* They provide inline documentation that integrates with tools like `mkdocstrings-python`, making the codebase more understandable and reusable.

### Best practices for documentation

- Update documentation when adding or changing features.
- Keep language clear and concise, avoiding jargon.
- Test documentation builds with `just docs-build` before committing.
- For code changes, ensure type hints and docstrings are complete—`just qa` will flag missing or incorrect ones.

---

## Development Workflow

Now that your environment is ready, here's the workflow for making changes:

1. **Create a feature branch**:

   ```sh
   git checkout -b name-of-your-bugfix-or-feature
   ```

   *Why a branch?* Branches isolate your changes, making code review and integration easier. Use descriptive names like `fix-weather-parsing-bug` or `add-humidity-forecast`.

2. **Make your changes** and write tests for new functionality.

3. **Before committing**, run quality assurance checks:

   ```sh
   just qa
   ```

   This command runs formatting, linting, type checking, and all tests. Fix any issues before proceeding.

4. **Commit with clear messages**:

   ```sh
   git add .
   git commit -m "Add feature: describe what changed

   - Explain the implementation approach
   - Reference any related issues: Fixes #123
   "
   ```

   *Why detailed messages?* Future maintainers (including yourself) will understand *why* a change was made, not just *what* changed.

5. **Push and submit a pull request**:

   ```sh
   git push origin name-of-your-bugfix-or-feature
   ```

   Then create a pull request on GitHub describing your changes and the problem they solve.

---

## Implementing a New Feature

This section provides a step-by-step guide for implementing a new feature, from planning to pushing your changes. It integrates the workflows covered in the previous sections (Environment Setup, Dependency Management, Testing & QA, Documentation, Git and Version Control).

### 1. Plan the Feature

- **Understand the requirements**: Review the issue or feature request. For example, if adding weather forecast downloading, clarify what data to fetch, how to handle errors, and where to store results.
- **Check existing code**: Look at similar features in the codebase to understand patterns and avoid duplication.
- **Estimate scope**: Keep the feature focused—small, incremental changes are easier to review and test.

### 2. Set Up Your Development Environment

- Ensure your environment is ready as described in the Environment Setup section.
- If the feature requires new dependencies, follow the Dependency Management workflow to add them to `pyproject.toml` and sync with `uv sync`.

### 3. Create a Feature Branch

- Follow the Git and Version Control workflow: Create a descriptive branch (e.g., `git checkout -b add-weather-forecast-download`).
- Work on this branch to keep `main` clean.

### 4. Implement the Code

- Write the code incrementally. For a weather forecast feature, start with a basic implementation using the Open-Meteo API.
- Add type hints and docstrings as required in the Documentation section.
- Follow the project's coding standards (checked by `just qa`).

### 5. Add and Run Tests

- Write tests for the new functionality as described in the Testing & QA section.
- Run `just test` to check your specific tests, then `just qa` to ensure everything passes (formatting, linting, type checking, and all tests).

### 6. Update Documentation

- Follow the documentation workflow: If the feature adds new functionality, update relevant files in `docs/` and ensure docstrings are complete.
- Preview changes with `just docs-serve` and build with `just docs-build`.

### 7. Commit Your Changes

- Stage changes with `git add -p` for focused commits.
- Write clear commit messages referencing issues (e.g., `git commit -m "Add weather forecast download feature\n\n- Implements API call to Open-Meteo\n- Fixes #123"`).
- Keep commits atomic and logical.

### 8. Push and Create a Pull Request

- Push your branch: `git push origin add-weather-forecast-download`.
- Create a pull request on GitHub, following the Pull Request Guidelines.
- Ensure CI checks pass (tests run automatically).

### 9. Iterate on Feedback

- Address review comments by making changes on your branch.
- Re-run `just qa` and tests before pushing updates.
- Once approved, your changes will be merged.

This workflow ensures features are implemented systematically, with quality checks at each step. For your weather forecast feature, start by integrating the provided API code into the project structure, then follow these steps to add tests, documentation, and submit a pull request.

---

## Git and Version Control

### Why Git matters for this project

Git captures the history of your changes and makes it possible to review, revert, and reuse work over time. For FAIR-compliant development, a clear version control workflow improves traceability and keeps the repository stable for collaborators.

### Basic workflow for every contribution

1. **Keep `main` clean**

   Work on a topic branch instead of committing directly to `main`.

   ```sh
   git checkout -b fix-forecast-label
   ```

2. **Review your work before staging**

   ```sh
   git status
   git diff
   ```

   Use `git diff` to confirm that the files changed match the feature or bug fix you intended.

3. **Stage logically related changes**

   ```sh
   git add -p
   ```

   This helps keep each commit focused. A small, single-purpose commit is easier to understand and easier to review.

4. **Write an atomic commit message**

   ```sh
   git commit -m "Fix calculation for daily forecast error"
   ```

   If the change needs more explanation, include a short body:

   ```sh
   git commit -m "Fix daily forecast error calculation"
   git commit --amend
   ```

   Good commit messages explain what changed and why. Avoid combining unrelated work in the same commit.

5. **Synchronize with the upstream repository**

   If you forked the project, add the original repository as `upstream` once:

   ```sh
   git remote add upstream https://github.com/sebastian-linden/fue.git
   git fetch upstream
   ```

   Then keep your branch current before pushing:

   ```sh
   git fetch upstream
   git rebase upstream/main
   ```

   If you are not using a fork, use `git pull --rebase` on the main repository instead.

6. **Push your branch to your fork**

   ```sh
   git push --set-upstream origin name-of-your-bugfix-or-feature
   ```

   Use the branch name consistently in your pull request.

### Best practices for traceability

- Use one branch per feature or bug fix.
- Keep commits small and focused.
- Include issue or task references in commit messages when applicable, for example: `Fixes #123`.
- Avoid rewriting published history after a branch has been shared. If you need to clean up commits before pushing, do so locally with `git rebase -i`.

### Checking your history and status

Run these commands to verify your work before creating a pull request:

```sh
git status
git log --oneline --decorate --graph --all
git diff --staged
```

These commands help you confirm that your final branch contains only the intended changes.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put your new functionality into a function with a docstring, and add the feature to the list in README.md.
3. The pull request should work for Python 3.12, 3.13, and 3.14. Tests run in GitHub Actions on every pull request to the main branch, make sure that the tests pass for all supported Python versions.

## Tips

To run a subset of tests:

```sh
uv run pytest tests/
```

## Releasing a New Version

1. **Bump the version** and **write the changelog:**
   ```bash
   uv version <version>        # or: uv version --bump minor
   ```
   Then write `CHANGELOG/<version>.md`. See previous entries for the format.
2. **Commit:**
   ```bash
   git add pyproject.toml uv.lock CHANGELOG/
   git commit -m "Release <version>"
   ```
3. **Release:**
   ```bash
   just release
   ```
   This creates an annotated `v*` tag, pushes it to GitHub, and creates a
   GitHub Release with the changelog contents as release notes. The tag
   push triggers `.github/workflows/publish.yml`, which builds the package,
   generates SLSA provenance attestations, and publishes to PyPI via
   trusted publishing.

## Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.
