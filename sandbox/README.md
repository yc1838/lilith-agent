# `run_python` Docker sandbox

Hardened image used by the `run_python` tool when `LILITH_SANDBOX=docker` (or `auto` with `docker` on PATH).

## Build

```bash
docker build -t lilith-pysandbox:latest sandbox/
```

Rebuild when you touch any file in this directory or change the library list in [Dockerfile](Dockerfile).

## Isolation summary

| Layer | Mechanism |
|---|---|
| Filesystem | `--read-only` rootfs + `--tmpfs /scratch:size=128m,noexec,nosuid,nodev` as CWD. No bind-mount of the host. |
| Network | `--network=bridge` (outbound allowed; needed for scraping). Container is in its own netns — host's `localhost` is unreachable. Cloud-metadata `169.254.169.254` blocked at Python socket layer via [sitecustomize.py](sitecustomize.py). |
| Privileges | `--cap-drop=ALL`, `--security-opt=no-new-privileges`, runs as UID 1000 (`sandbox` user). |
| Resource | `--memory=512m --memory-swap=512m --cpus=1.0 --pids-limit=128`; inside Python, rlimits on CPU, address space, and max file size. |
| Secrets | Host env is never forwarded — no `--env`/`-e` flags. The runner reads user code from stdin so nothing sensitive lands in `argv`. |

## What this does NOT stop

- LLM-generated code that calls `libc.connect(2)` via `ctypes` — the Python-layer metadata-IP block is a sitecustomize monkey-patch on `socket.getaddrinfo` / `socket.create_connection`. Bypassable by anyone who really wants to; the rest of the container still contains them.
- Outbound connections to public internet addresses. If you need a DNS allowlist, run with `--dns=<your_resolver>` and point it at a filtering resolver.
- Resource exhaustion at the Docker daemon level (cold-start latency, image pulls). Pre-build the image and keep it warm.
