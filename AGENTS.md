# lib-python-worktree -- agent guide

Git-worktree lifecycle + contract engine, being extracted from the
`agent-worktree` MCP plugin. README.md covers what it does and how to use it;
`pyproject.toml` and `.github/workflows/` are the source of truth for structure,
testing, and release. This file records only the non-obvious invariants a
contributor must not silently break.

## Status: empty frame

This repo is currently the skeleton only -- `src/lib_python_worktree/` exposes
just `__version__`. The engine modules (worktree lifecycle, the `.seretos/`
setup-contract schema + loader, port allocation, the setup-script runner, the
pluggable state store) are migrated from `agent-worktree` in a follow-up,
driven by the first release ticket. Until then there is no public API beyond
the version.

## Layering (read before grounding a change)

- The engine lives here, under `src/lib_python_worktree/`. It is registry- and
  filesystem-aware but **MCP-agnostic**: no `mcp` import belongs in this
  package, and engine functions return plain dicts/lists/dataclasses.
- The `agent-worktree` plugin is a **separate repo** and only wraps the engine
  as `@mcp.tool()`s (`worktree_create`/`worktree_list`/`worktree_remove`).
  Behaviour, the git subprocess handling, the contract data model, and the
  setup runner are changed **here**, not in the plugin. The MCP tool docstrings
  (the LLM-facing descriptions) live in the plugin.

## Repo specifics (minimal by design)

- **Language:** Python, src-layout under `src/`, package `lib_python_worktree`.
- **Tests:** `python -m pytest`. Install dev deps with
  `pip install -e ".[test]"`.
- **Branch discipline:** All feature work happens on a feature branch in a git
  worktree, never on `main`. Assume the worktree and branch already exist and
  that you are inside them.
- **AI attribution:** The project-issues MCP automatically prefixes every
  comment and PR body with `#ai-generated`. Never type that prefix yourself.

## Release is pipeline-owned

`release.yml` (manual dispatch, `version=X.Y.Z`) stamps the version in CI, tags
`vX.Y.Z`, force-pushes `release/Nx`, publishes a GitHub Release, then opens a
dependency-update ticket in the consumer (`Seretos/agent-worktree`). Never
hand-bump `version` in `pyproject.toml`.

The ticket step authenticates with the **`WORKTREE_TICKET_TOKEN`** repo secret
(a fine-grained PAT with **Issues: write** on `Seretos/agent-worktree`);
`GITHUB_TOKEN` cannot open cross-repo issues. The step is
`continue-on-error: true`, so a missing or invalid token never blocks the
release itself.

**If the automatic step was skipped or failed**, re-file manually by running
the `open-dep-ticket` workflow (`.github/workflows/ticket.yml`) via "Run
workflow" in GitHub Actions. Supply:

- `version` -- the semver string (no leading `v`), e.g. `0.2.0`.
- `consumers` -- space-separated `owner/repo` targets (default:
  `Seretos/agent-worktree`).

The workflow is idempotent: it checks for an open issue with the exact same
title before creating one, so running it twice is safe.

**Human prerequisite -- `WORKTREE_TICKET_TOKEN`:** create this repository secret
(Settings -> Secrets -> Actions) once before the first release.
