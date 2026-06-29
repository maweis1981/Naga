# Security Policy

## Scope

Naga is designed primarily for local development on Apple Silicon machines. Some features, especially document ingestion, MCP server management, and admin endpoints, are intentionally powerful and should be treated as privileged operations.

## Supported Usage

- Prefer binding the server to `127.0.0.1`.
- If you expose the server beyond localhost, set `NAGA_ADMIN_TOKEN` and avoid sharing that token.
- Only configure MCP servers you trust.
- Only ingest documents from paths you trust.

## Reporting a Vulnerability

Please report suspected security issues privately to the maintainer before opening a public issue. Include:

- affected version or commit
- reproduction steps
- impact assessment
- any suggested mitigation

## Known Risk Areas

- `/admin/*` endpoints can change models, add MCP tools, and index local files.
- MCP servers run as local subprocesses with the current user privileges.
- Vision and document helpers can fetch or read user-supplied paths and URLs.
