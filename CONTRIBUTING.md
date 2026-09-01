# Contributing

Thanks for contributing to pythermalcomfort.

## Workflow

- All changes land via a pull request — direct pushes to `master` or
  `development` are blocked for everyone except the repository admin.
- Open PRs against `development`, not `master`. `master` reflects the
  latest released version.
- A pull request needs:
  - **Approval from a code owner** (see `.github/CODEOWNERS`) before it can
    be merged. Approving your own PR, or another contributor's PR, does not
    satisfy this — it must come from the code owner.
  - The `format` and `test` checks (`.github/workflows/pull-request.yml`)
    passing — ruff lint/format and the tox test suite must be green.
  - All review comment threads resolved.
- Keep PRs focused and reasonably small — it makes review faster.

## Before opening a PR

- Run `ruff check` / `ruff format` and `tox` locally so CI doesn't surprise
  you.
- Rebase or merge the latest `development` into your branch if it's fallen
  behind, to avoid unnecessary conflicts at review time.

## Merging

Only the repository admin merges pull requests. If your PR is approved and
CI is green, it will be merged for you.
