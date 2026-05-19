# Buckyball Docker Sandbox

```bash
# Build image (requires Nix)
python scripts/docker/build.py

# Run container (bind-mounts current repo, exposes bbdev API on port 3000)
python scripts/docker/run.py [--repo PATH] [--port PORT] [--name NAME]

# Multi-agent: one worktree per agent
git worktree add ../bb-agent-1 main
python scripts/docker/run.py --repo ../bb-agent-1 --port 3001
```
