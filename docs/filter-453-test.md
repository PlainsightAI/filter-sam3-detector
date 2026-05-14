# FILTER-453 Schema-Emit Test Runbook

> **Scope.** This branch (`feat/filter-453-test`) exercises [PlainsightAI/gh-actions PR #82](https://github.com/PlainsightAI/gh-actions/pull/82) — the FILTER-453 schema emit + OCI-referrer attach work — against `filter-sam3-detector` ahead of the PR merging. The branch is a throwaway test harness, not destined for `main`. Delete when FILTER-453 is merged and `openfilter` v0.2.0 has shipped.
>
> **Tickets.** [FILTER-453](https://plainsight-ai.atlassian.net/browse/FILTER-453) (gh-actions PR under test), [FILTER-443](https://plainsight-ai.atlassian.net/browse/FILTER-443) (stub migration tracking this filter).

## What this test does

Three pieces of scaffolding land on this branch:

1. **`filter_sam3_detector/schema.py`** — stub `FilterConfigBase` + `FilterOutputSchema` subclasses. `filter.py` (legacy dict-like `FilterConfig`) is untouched. The stub gives `openfilter emit-schema` something to discover. The real FILTER-443 migration will port `FilterSAM3DetectorConfig` and delete the stub.

2. **`pyproject.toml`** — relaxes the `openfilter[all]` upper bound to `<0.3.0` and adds `[tool.uv.sources]` pinning to a commit on `openfilter@main`. `FilterConfigBase` and `openfilter.cli.cmd_emit_schema` only exist there; no released `openfilter` ships them yet.

3. **`.github/workflows/test-filter-453.yaml`** — `workflow_dispatch`-only side workflow. Builds the image, runs `emit-schema` inside it (dry-run path), and uploads the emitted JSON as a build artifact for inspection.

The workflow **bypasses the reusable workflow** (`filter-release-private.yaml`) and pins the `publish-gar-image` composite directly. See [Known flaw #2](#known-flaws-in-filter-453-as-of-test-date) for why.

## Reproducing the run

### Prerequisites

- The branch `feat/filter-453-test` exists on the remote with the test scaffolding committed.
- Repo secrets needed:
  - `HF_TOKEN` — gated SAM3 weights download (repo-level, already provisioned).
  - `GH_BOT_USER_PAT` — checks out the private `PlainsightAI/gh-actions` repo at the PR ref (org-level, already provisioned per `apply-rulesets.yaml`).
- `gh` CLI authenticated with read access to this repo and PR #82's metadata.

### Run the workflow

```bash
gh workflow run test-filter-453.yaml --ref feat/filter-453-test
gh run list --workflow=test-filter-453.yaml --limit 1
```

Grab the run ID from the listing, then either wait or watch:

```bash
RUN_ID=<id-from-list>
gh run watch "$RUN_ID" --exit-status        # blocks until done; non-zero exit on failure
```

Cold-runner build profile is ~3–4 min on `ubuntu-latest`: 2 min for the Docker build, ~50s for the SAM3 weights pull, ~10s for the emit-schema steps. The PR's 60s `docker run` timeout is comfortable for this filter.

### Inspect the emitted schemas

```bash
RUN_ID=<id-from-list>
gh run download "$RUN_ID" -D /tmp/filter-453-out
jq . /tmp/filter-453-out/filter-453-schemas/config-schema.json | less
jq . /tmp/filter-453-out/filter-453-schemas/output-schema.json | less
```

The healthy outputs:

- `config-schema.json` ≈ 3.3 KB, 21 properties (stub fields + inherited operator-facing fields from `FilterConfigBase`: `exit_after`, `batch_size`, `accumulate_timeout_ms`). Managed fields (`sources`, `outputs`, `log_path`, …) excluded by default.
- `output-schema.json` ≈ 2.0 KB, top-level `SAM3DetectorOutput` with `$id`, `x-openfilter-frame-data-key: ""` (whole-namespace anchor), and `SAM3Detection` nested in `$defs` with its own `$id`. `detections` array items resolved via `$ref`.

## Known flaws in FILTER-453 (as of test date)

These are gating issues to feed back to PR #82 before it merges. The workflow on this branch already works around #2 and #4 so the test can actually run.

1. **`openfilter` version mismatch.** `FilterConfigBase` / `FilterOutputSchema` / `openfilter.cli.cmd_emit_schema` exist only on `openfilter@main` (unreleased v0.2.0). Released versions (≤ v0.1.30) lack all three. Without a `[tool.uv.sources]` override, the action's `docker run --entrypoint python … -m openfilter.cli emit-schema` fails inside the image with `ModuleNotFoundError`. PR #82 should either gate on v0.2.0 shipping to the index or document the override.

2. **Workflow + composite ref incoherence.** `filter-release-{private,premium}.yaml` at the feat ref still call `uses: PlainsightAI/gh-actions/publish-gar-image@main`. The new emit/attach logic lives on the feat branch of the composite, *not* on main. Composite actions silently drop unknown inputs, so calling the reusable workflow at the feat ref *succeeds* without ever running schema steps — a false-pass. PR #82 should pin the composite to the same ref (commit SHA preferred) so the workflow and composite are coherent pre-merge.

3. **Public-workflow not patched.** PR #82 only modifies the *private* `PlainsightAI/gh-actions` repo. The public `gh-actions-public/filter-release.yaml` — which this repo uses for Docker Hub releases per [FILTER-428](https://plainsight-ai.atlassian.net/browse/FILTER-428) — has no `schema_module` input. Schema transport won't reach the public release path without a parallel patch.

4. **Cross-repo action access.** `PlainsightAI/gh-actions` is private. The default `GITHUB_TOKEN` of an adopting filter's workflow has no read access, so `uses: PlainsightAI/gh-actions/...@<ref>` fails at job setup with `Unable to resolve action … not found`. The fix is either (a) configure org-level "actions access" so adopting filter repos can fetch from `gh-actions`, or (b) the manual-checkout pattern this workflow uses (check out the action repo under a PAT, consume it as a `./local/path` action). This isn't strictly a PR #82 bug, but consumers will hit it first if they don't know.

## After PR #82 merges

When FILTER-453 lands and `openfilter` v0.2.0 ships to the index, this branch is no longer load-bearing. To wind it down:

1. Delete `filter_sam3_detector/schema.py` (FILTER-443 will land the real port to `FilterConfigBase` in `filter.py`).
2. Revert `pyproject.toml`: drop the `[tool.uv.sources]` `openfilter` entry, restore the upper bound to `<0.3.0` or `<0.2.0` as appropriate for the actual released version.
3. Delete `.github/workflows/test-filter-453.yaml`.
4. Delete this file.
5. Delete the branch on origin: `git push origin --delete feat/filter-453-test`.

If FILTER-453 evolves and a follow-up needs testing, re-derive a new throwaway branch from these notes rather than reusing this one.

## Tips for a fresh agent picking this up

- **Don't push a `v0.2.0` tag on openfilter** to make the dep resolve. The release pipeline may publish to PyPI, which permanently burns the version even after tag/release deletion. The `[tool.uv.sources]` override is the right escape hatch.
- **Don't open a PR** for this branch. The workflow is `workflow_dispatch`-only and the branch is throwaway. `gh workflow run --ref feat/filter-453-test` works against any pushed branch.
- **If the dispatch fails at parse time** with `Unrecognized named-value: 'inputs'`, you've put `${{ inputs.* }}` in a `uses:` field. GitHub Actions disallows it; `uses:` must be a literal string.
- **If the dispatch fails with `Unable to resolve action … not found`**, the `GH_BOT_USER_PAT` secret is missing or the manual-checkout step was edited out. Re-add the `Checkout gh-actions at the PR ref` step.
- **If `make build-image` fails on `HF_TOKEN`**, the secret-forwarding step at the top of the workflow guards against this — check `HF_TOKEN` is set at the repo or org level.
- **If schema emit fails with `ModuleNotFoundError: openfilter.cli.cmd_emit_schema`**, the `[tool.uv.sources]` override didn't apply. Confirm the Dockerfile's install path uses `uv` (it does — `uv pip install --system -e .`). `pip` would ignore the override.
- The composite's 60s `docker run` timeout on emit-schema was *not* tripped by this filter's heavy imports (torch/transformers/sam3). Each emit ran in 4–6 s. Don't preemptively raise the timeout.
