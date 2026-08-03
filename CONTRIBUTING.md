# Contributing

These are the standards every exercise submission follows. They're deliberately the same standards used on real tickets, so getting them right here carries over directly.

## Git and branching

We use trunk-based development off `main`. `main` is always deployable. Never commit to it directly.

### Branch naming

```
<type>/<short-description>
```

| Type | Use for | Example |
|---|---|---|
| `feature` | New functionality | `feature/nn-xor` |
| `fix` | Bug fixes | `fix/nn-gradient-bug` |
| `chore` | Tooling, config, docs | `chore/update-readme` |

For an exercise, branch as `feature/<exercise-name>-<yourname>`, e.g. `feature/nn-xor-yourname`.

### Workflow

```bash
git checkout main
git pull origin main
git checkout -b feature/nn-xor-yourname
# work, commit as you go
git push -u origin feature/nn-xor-yourname
# open a Pull Request into main
```

- Keep the branch scoped to the exercise only. No unrelated changes.
- Rebase on `main` (do not merge `main` into your branch) if it drifts behind before you open the PR.
- Delete the branch after merge.

## Commit messages

We follow [Conventional Commits](https://www.conventionalcommits.org/). Every commit message has the form:

```
<type>(<scope>): <summary>

[optional body]

[optional footer]
```

| Type | Meaning |
|---|---|
| `feat` | A new feature |
| `fix` | A bug fix |
| `test` | Adding or correcting tests |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `docs` | Documentation only |
| `chore` | Build process, tooling, dependencies |

### Good examples

```
feat(nn-xor): implement forward pass
feat(nn-xor): implement backpropagation and training loop
test(nn-xor): add truth table coverage for trained network
docs(nn-xor): record final architecture and loss
```

### Rules of thumb

- Imperative mood: `add tests`, not `added tests` or `adds tests`.
- Keep the summary line under about 72 characters.
- One logical change per commit. Do not squash the whole exercise into a single final commit — we want to see your process.
- No `WIP`, `fix2`, `asdf` style messages in the final history.

## Development environment

### Python setup

- Use a conda environment per project. Do not install packages globally.
- Pin dependency versions in `requirements.txt`.

```bash
conda create -n nn-xor python=3.10
conda activate nn-xor
pip install -r requirements.txt
pytest
```

### VS Code

- Install VS Code and open the repository as a folder, not individual files, so workspace settings and debugging work correctly.
- Install the Python extension and the Pylance language server.
- Point VS Code at your virtual environment interpreter (Command Palette → `Python: Select Interpreter`).
- Install a formatter (Black) and a linter (Ruff or Flake8), and enable format on save.
- Use the built-in integrated terminal for git, pip, and pytest commands.

### Claude Code

Claude Code (or any AI coding assistant) is available to help you write and review code faster. A few ground rules:

- You own everything you submit. If it writes something, you must be able to explain every line in review.
- Use it to scaffold files, explain unfamiliar APIs, and write boilerplate tests — not to skip understanding the core problem. The point of each exercise is that you can derive and reason about the solution yourself.
- Review and edit generated code before committing. Do not paste and commit without reading it.
- Prefer small, specific prompts (e.g. "write a pytest test for the trained network's XOR predictions") over "write the whole thing for me."
- State in your PR description if an AI assistant materially helped with a part of the solution. This is normal and expected — just be clear about it.

### Toolchain checklist

- Git installed and configured (`git config --global user.name` / `user.email`).
- Language runtime installed matching the exercise's stated version.
- Repository forked, cloned, and running locally, with dependencies installed.
- Existing test suite runs and passes before you start.

## Pull request checklist

Before requesting review, confirm:

- Branch named per the convention above and up to date with `main`.
- Commit history follows Conventional Commits and tells a clear story.
- Test suite passes locally with no failing or skipped tests.
- The exercise's own acceptance criteria (in its README) are met.
- Reviewer assigned.

Ask questions early rather than guessing for hours. Getting unblocked fast matters more than looking like you never needed help.
