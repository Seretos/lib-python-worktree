# Example: Docker-Compose Webapp Worktree

## Use-case

This contract models a full-stack web application that is isolated in its own
Docker Compose stack. Each worktree gets dedicated containers, so parallel
feature branches never share a database, a dev server port, or a running
migration.

`isolation: full` means the setup runner will execute the `setup` steps when
the worktree is created and the `teardown` steps when it is removed. Port slots
`app` and `chrome` are allocated from the global port range so there are no
collisions across concurrent worktrees.

## Prerequisites

- **Docker Desktop** (or any Docker Engine with the Compose v2 plugin) must be
  installed and the daemon must be running before the setup runner executes.
- **Node.js** (LTS) with **pnpm** available on `PATH`.
- **Prisma CLI** installed as a dev-dependency in `package.json` (or globally
  via `pnpm add -g prisma`).

## What each step does

### Setup

1. `start services` — `docker compose up -d`  
   Starts all services declared in `compose.yml` (database, cache, etc.) in
   detached mode. The setup runner waits for this step to exit before moving
   on.

2. `install deps` — `pnpm install`  
   Installs Node dependencies from the lockfile into the worktree's local
   `node_modules`. Running this per-worktree ensures each branch can have
   independent package overrides without affecting others.

3. `run migrations` — `pnpm prisma migrate dev`  
   Applies any pending Prisma migrations to the worktree's dedicated database
   container. Must run after `start services` because it needs the database to
   be reachable.

### Teardown

1. `stop services` — `docker compose down`  
   Stops and removes all containers for this worktree's stack, releasing their
   ports and network resources.

2. `remove containers` — `docker compose rm -f`  
   Force-removes any stopped containers that `down` may have left behind (e.g.
   if the first step partially failed).

## Port slots

| Slot     | Typical use                          |
|----------|--------------------------------------|
| `app`    | The Next.js / Vite dev server port   |
| `chrome` | Playwright's remote debugging port   |

Port numbers are assigned at worktree-creation time by the W4 allocator and
injected into the environment so processes bind to the right port without
hard-coding anything.

## Known limitations

- `docker compose` (v2 plugin syntax, not the legacy `docker-compose`) must be
  on `PATH`. If you are on Docker Desktop for Mac/Windows the plugin is bundled;
  on Linux you may need `apt install docker-compose-plugin`.
- The Docker daemon must already be running before the setup runner starts. If
  it is not, `docker compose up -d` will fail and the entire setup will abort.
- `pnpm prisma migrate dev` is interactive by default. Make sure your
  `compose.yml` includes a `healthcheck` on the database service so that Prisma
  does not race against an initialising container.
