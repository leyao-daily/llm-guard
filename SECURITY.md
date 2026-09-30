# Security policy

## Reporting a vulnerability

This gateway sits in the request path of production LLM traffic and holds upstream
credentials in memory. A vulnerability here is therefore worth reporting privately
first.

**Please email `yaole.intel@gmail.com`** rather than opening a public issue.

Include, as far as you can:

- the version or commit you tested,
- a minimal reproduction (a `curl` against a local instance is ideal),
- the impact you believe it has,
- whether you intend to publish.

**What to expect:** an acknowledgement within 72 hours, and a coordinated
disclosure timeline of 90 days or shorter by mutual agreement. Credit in the
release notes unless you prefer otherwise.

## Scope

In scope:

- The proxy request/response path (`llmguard/gateway.py`).
- Credential handling — keys must never be logged, stored, or forwarded to a
  different upstream than the one they were issued for.
- Budget and detection bypasses: any input that causes a configured limit to be
  exceeded, or a detector to be evaded.
- SQL injection, path traversal, or header injection through proxied input.
- Denial of service against the gateway itself (unbounded memory, body size,
  connection exhaustion).

Out of scope:

- Vulnerabilities in upstream providers.
- Misconfiguration by the operator (for example `--no-verify-upstream-tls`, which
  logs a warning by design).
- Running the gateway directly on the public internet without TLS termination in
  front of it. Inbound TLS termination is deliberately out of scope for this
  project; put a load balancer or reverse proxy in front.

## Design decisions that matter to security reviewers

These are deliberate, and worth understanding before reporting them as issues:

1. **Zero third-party runtime dependencies.** The package imports only the Python
   standard library. There is no `pip install`, so there is no upstream package
   that can be compromised in order to compromise the gateway. This is a direct
   response to the 2026-03 LiteLLM PyPI supply-chain compromise
   (`litellm_init.pth` executing on every interpreter start).

2. **Credentials are never persisted.** The gateway stores a truncated SHA-256
   fingerprint of a caller's token, never the token. Upstream credentials are held
   in memory only.

3. **Prompts and completions are never stored.** Only token counts, cost, latency,
   status and attribution labels are written. This keeps the operator out of the
   PII path.

4. **Outbound TLS is verified by default.** `--no-verify-upstream-tls` exists but
   logs a warning; the supported answer to a corporate MITM proxy is
   `--upstream-ca`.

5. **Budgets fail closed, detection fails open.** A budget with `action=block`
   returns 429 once exceeded. Detectors only ever *report* — a security control
   that silently kills production traffic is a worse failure than the overspend it
   prevents.

6. **Request bodies are bounded** (32 MiB) and so is the accounting buffer, which
   falls back to a synchronous write rather than dropping records.
