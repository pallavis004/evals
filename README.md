# evals

Practice exercises for VitalView AI interns. This is where new engineers get evaluated on their first tasks: small, scoped exercises that mirror how we actually work, not toy problems disconnected from real practice.

## What this is

Each exercise lives under [`exercises/`](exercises/) with its own README describing the objective, constraints, and acceptance criteria. The code itself is usually small. The point is the process: branching, commit hygiene, tests, and a PR that tells a clear story.

## How to submit a solution

We use a fork + pull request workflow:

1. Fork this repository.
2. Clone your fork and follow the branching and commit standards in [`CONTRIBUTING.md`](CONTRIBUTING.md).
3. Work the exercise in `solutions/<your-name>/<exercise-name>/` in your fork.
4. Push your branch and open a pull request back into `VitalView-AI/evals` `main`.
5. Request a review.

Solutions are reviewed like real PRs. Once merged, they stay in the repo so other interns can see different approaches.

## Exercises

1. [`exercises/nn-xor`](exercises/nn-xor/README.md) — implement a feedforward neural network from scratch in NumPy that learns XOR.
2. [`exercises/mnist-classifier`](exercises/mnist-classifier/README.md) — extend it to a multi-class handwritten digit classifier (softmax, mini-batch SGD, train/test split).

## Ground rules

- Keep your branch scoped to the exercise only.
- AI coding assistants (Claude Code, etc.) are welcome, but you own everything you submit. If a reviewer asks you to explain a line, you should be able to.
- See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full git, commit, and environment standards.
