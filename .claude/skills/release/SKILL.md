---
name: release
description: Cut a pythermalcomfort release - RC to TestPyPI from development, or a final version to PyPI from master. Use when asked to release, publish, cut an RC, bump the version, or ship a new version of the package.
---

# Releasing pythermalcomfort

Two-stage process: an RC from `development` publishes to **TestPyPI**, then a final
tag from `master` publishes to **PyPI**. Both are triggered by pushing a git tag.

`bump-my-version` auto-commits and auto-tags (`commit = true`, `tag = true`), and
picks between the `Xrc{N}` and plain `X.Y.Z` formats on its own via `pre_n`'s
`optional_value`. In the normal case you do not touch git yourself after running
it — just push.

Always invoke it through the project environment: `pipenv run bump-my-version`.

## Guardrails - read before starting

**CI does not catch a final tag cut from the wrong branch.** The RC workflow
verifies its tag is on `development` (`git merge-base --is-ancestor`). The final
workflow has **no equivalent check for `master`** — it triggers on `v*` and gates
only on the name not containing `rc`. A final tag pushed from `development` will
publish to PyPI regardless. Verify the branch yourself at step 7.

**`allow_dirty = false`.** Any uncommitted change aborts the bump. Check
`git status` is clean before every bump.

**Never re-tag or force-push a published version.** PyPI rejects re-uploads of an
existing version. If a release is broken, bump to a new patch version instead.

**Do not run any step's command until the previous step's verification passed.**
Each gate below exists because skipping it produces a broken or unpublishable
release.

## 0. Preflight

Confirm the validation-data pin is current. `tests/conftest.py` pins
`unit_test_data_prefix` to a tag of `validation-data-comfort-models`, not `main`.

```bash
# current pin
grep -o 'validation-data-comfort-models/[^/]*' tests/conftest.py
# latest available tag
git ls-remote --tags --refs https://github.com/FedericoTartarini/validation-data-comfort-models.git \
  | sed 's#.*refs/tags/##' | sort -V | tail -3
```

If the pin is behind, read that repo's `CHANGELOG.md`, bump the pin, and re-run the
affected tests **before** continuing. Do not ship a release pointed at a stale tag.

Then confirm the tree is healthy:

```bash
git checkout development && git pull
git status --short          # must be empty
pipenv run pytest tests/ -x -q
pipenv run ruff format --check ./pythermalcomfort ./tests
pipenv run ruff check ./pythermalcomfort ./tests
```

**Gate:** tests pass, format and lint clean, working tree empty.

## 1. Update CHANGELOG.rst

Add an entry covering every user-facing change since the last release. Get the
range with:

```bash
git log --oneline "$(git describe --tags --abbrev=0)"..HEAD
```

Commit it on its own (`docs: add CHANGELOG entry for X.Y.Z`). Ask the user before
committing.

**Gate:** `CHANGELOG.rst` describes this release and is committed.

## 2. Cut the RC (publishes to TestPyPI)

On `development`. Pick the part that reflects the change; it lands on `X.Y.Zrc1`
automatically.

```bash
pipenv run bump-my-version bump minor    # or patch / major
git push origin development --tags
```

If `bump-my-version` fails to commit because a pre-commit hook (e.g. `ruff format`)
modified a file mid-commit, recover manually — do not re-run the bump, which would
double-increment:

```bash
git add <reformatted-file>
git commit -m "Bump version: A.B.C → X.Y.Z"   # exact message bump-my-version printed
git tag vX.Y.Z -m "Bump version: A.B.C → X.Y.Z"
```

**Gate:** tag `vX.Y.Zrc1` exists and is pushed, and `git merge-base --is-ancestor
<tag-sha> origin/development` succeeds — CI enforces this and the push fails
otherwise.

## 3. Verify the RC

Confirm the `deploy-testpypi` job passed and the package is on TestPyPI.

```bash
gh run list --limit 5
```

**Gate:** green CI and the version visible on TestPyPI. Do not proceed on a red or
still-running build.

## 4. Need another RC?

Only increment the RC number — never re-tag an existing one:

```bash
pipenv run bump-my-version bump pre_n
git push origin development --tags
```

`pre_n` **only works when the current version is already an RC.** Run it on a plain
version and it aborts with:

```
ValueError: The given value 0 is lower than the first value 1 and cannot be bumped.
```

That means you are not mid-RC — go to step 2 instead.

Do **not** reach for `bump minor`/`patch` here. From `4.5.0rc1` that yields
`4.6.0rc1`, silently skipping 4.5.0 altogether.

Then return to step 3.

## 5. Merge development into master

Open a PR from `development` to `master` and confirm CI passes before merging.

```bash
gh pr create --base master --head development --title "Release X.Y.Z"
```

**Gate:** PR merged and CI green on `master`.

## 6. Check out master

```bash
git checkout master && git pull
```

**Gate:** `git rev-parse --abbrev-ref HEAD` prints `master`, and `git status` is
clean.

## 7. Finalize the release (publishes to PyPI)

Dropping the RC suffix is not a single-word part bump, so pass the target
explicitly.

**Verify the branch first — CI will not do it for you:**

```bash
test "$(git rev-parse --abbrev-ref HEAD)" = master || echo "STOP: not on master"
pipenv run bump-my-version bump --new-version X.Y.Z
git push origin master --tags
```

**Gate:** tag `vX.Y.Z` (no `rc`) pushed from `master`.

## 8. Confirm publication

Check the `Test and publish pythermalcomfort` action succeeded and the version is
live on PyPI.

**Gate:** green action, package installable.

## 9. Sync master back into development

Do not skip this — the version-bump commit lives only on `master` until you do, and
the next RC will be cut from a stale base.

```bash
git checkout development && git pull
git merge origin/master --no-edit
git push origin development
```

**Gate:** `git log origin/development -1` includes the bump commit.

## Quick reference

| Stage | Branch | Tag | Publishes to | Branch checked by CI |
|---|---|---|---|---|
| RC | `development` | `vX.Y.ZrcN` | TestPyPI | yes (`merge-base`) |
| Final | `master` | `vX.Y.Z` | PyPI | **no — verify yourself** |

Version progression (verified against this repo's `.bumpversion.toml`):

| From | Command | To |
|---|---|---|
| `4.4.2` | `bump patch` | `4.4.3rc1` |
| `4.4.2` | `bump minor` | `4.5.0rc1` |
| `4.5.0rc1` | `bump pre_n` | `4.5.0rc2` |
| `4.5.0rc2` | `bump --new-version 4.5.0` | `4.5.0` |
| `4.5.0rc1` | `bump minor` | `4.6.0rc1` ← skips 4.5.0, almost never what you want |

Check any bump before running it for real:

```bash
pipenv run bump-my-version bump <part> --dry-run -v | grep -i "new version"
```
