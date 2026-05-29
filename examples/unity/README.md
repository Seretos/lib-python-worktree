# Example: Unity Project Worktree

## Use-case

This contract models a Unity project where every worktree shares the host
machine's Unity Hub installation and Editor. There are no containerised
services, no port-bound dev servers, and no dependency-install step — the
Editor itself manages asset import and compilation when it opens the project.

`isolation: none` is the right choice here because:

- Unity projects rely on a single licensed Editor binary installed by Unity Hub
  on the host. Running separate Editor instances per worktree is possible but
  they all share the same Hub and the same licence activation — there is nothing
  to isolate.
- There is no network service to start or stop, so `setup`/`teardown` steps
  would be empty anyway.
- Port allocation is unnecessary because the Editor's local HTTP server (used
  by the Editor for the profiler, remote settings, etc.) is started on demand
  by Unity itself, not by a setup script.

## Prerequisites

- **Unity Hub** installed on the host with a valid licence.
- The exact **Unity Editor version** pinned in `ProjectSettings/ProjectVersion.txt`
  must be installed through Unity Hub before the worktree is opened.
- No additional CLI tools are required for the contract itself; build scripts
  (if any) live in `Assets/` or a `Makefile` and are invoked manually.

## Known limitations

- **No port allocation:** `isolation: none` contracts do not support `ports`
  fields. If your workflow requires a dedicated port (e.g. for a local game
  server), upgrade `isolation` to `full` or `partial` and add the appropriate
  setup steps.
- **No setup/teardown:** Automated pre/post-worktree scripts are unavailable
  under `isolation: none`. Any one-time environment setup (e.g. installing a
  Unity package from a private registry) must be done manually or via a
  separate script invoked outside the contract runner.
- **Shared asset cache:** Unity Hub's global package and asset caches are
  shared across all worktrees on the host. Parallel branches that modify
  `Packages/manifest.json` may trigger simultaneous cache writes — open only
  one Editor per branch at a time to avoid conflicts.
