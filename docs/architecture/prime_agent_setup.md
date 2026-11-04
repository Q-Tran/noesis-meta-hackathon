# Prime Agent setup on this machine

What it took to run Prime Agent behind AT&T's TLS-inspecting proxy with GitHub
Copilot and Databricks models. Every item here was found by running the thing
and reading the actual error, not from documentation, so it is worth recording.

## Install

`npm i -g prime-agent` does not work, for two independent reasons:

- **There is no `prime-agent` package on the public npm registry** (`npm view`
  returns 404). The docs say so obliquely: "The inherited npm workspace
  identifiers in the source tree are not the public install path."
- npm's global prefix defaults to `/usr`, which is root-owned, so any `-g`
  install fails with `EACCES ... mkdir '/usr/lib/node_modules/...'`.

```bash
npm config set prefix ~/.local        # ensure ~/.local/bin is on PATH
curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh
```

The official installer also ends in `npm install -g`, so the prefix fix is
required either way.

## The one that masks everything else: NODE_EXTRA_CA_CERTS

The proxy performs TLS interception. `curl`, Python, and Go trust its CA through
`SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`, which are already set in `~/.profile`.
**Node reads neither.** Without its own variable, every HTTPS call from Node
fails with `UNABLE_TO_GET_ISSUER_CERT_LOCALLY`, which Prime Agent reports as a
bare `Connection error.` -- indistinguishable from a wrong URL or a bad key.

```bash
# ~/.profile, beside the other CA variables
export NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
```

This was the root cause of *both* providers failing. Diagnosing it needed a
throwaway extension that wrapped `globalThis.fetch` and logged `e.cause.code`;
nothing in Prime's own output named it.

## GitHub Copilot

The built-in `github-copilot` provider cannot work on this tenant:

- `POST api.github.com/copilot_internal/v2/token` (the token exchange) returns
  **403** from this network, with GitHub's anti-scraping body.
- A `gho_` OAuth token carries no embedded `proxy-ep=`, so Prime's
  `getGitHubCopilotBaseUrl()` falls back to `api.individual.githubcopilot.com`
  -- the consumer host, which this Copilot Business tenant is not served on
  (`ERR_HTTP2_STREAM_ERROR`).

The token *is* accepted directly by `api.githubcopilot.com`, so it is registered
as a plain custom provider instead. A separate provider id matters: the built-in
provider's `modifyModels()` recomputes `baseUrl` from the token, so overriding
`baseUrl` on `github-copilot` is silently discarded.

```jsonc
// ~/.prime/agent/models.json
"copilot": {
  "baseUrl": "https://api.githubcopilot.com",
  "api": "openai-completions",
  "apiKey": "copilot",          // resolved from auth.json, not a literal
  "authHeader": true,
  "headers": {
    "Copilot-Integration-Id": "vscode-chat",
    "Editor-Version": "vscode/1.136.1",
    "Editor-Plugin-Version": "copilot-chat/0.48.1",
    "User-Agent": "GitHubCopilotChat/0.48.1"
  },
  "models": [ /* claude-opus-5, claude-sonnet-5, claude-opus-4.8, claude-haiku-4.5 */ ]
}
```

`gpt-5.5` and `gpt-6-astra` are deliberately absent: this host serves them only
on `/responses`, and they 400 on `/chat/completions`.

## Databricks

Ported from `~/.config/opencode/plugins/databricks_plugin.js`. Three separate
incompatibilities, and it matters which layer fixes which.

**Request shape -- fixed by config.** The gateway *rejects* unknown fields
rather than ignoring them, one at a time:

| Error | Flag |
|---|---|
| `store: Extra inputs are not permitted` | `compat.supportsStore: false` |
| `tools.0.custom.strict: Extra inputs are not permitted` | `compat.supportsStrictMode: false` |

```jsonc
"databricks": {
  "baseUrl": "https://adb-....azuredatabricks.net/ai-gateway/mlflow/v1",
  "api": "openai-completions",
  "apiKey": "DATABRICKS_TOKEN",
  "authHeader": true,
  "compat": { "supportsStore": false, "supportsStrictMode": false }
}
```

The second one only appears **with tools enabled**, which is every real session
and no `-nt` smoke test. A print-mode test with `-nt` passes while interactive
use fails with `400 status code (no body)` -- so verify with tools on.

**Response shape -- needs an extension.** Databricks returns OpenAI-shaped SSE
except that `choices[].delta.content` is an *array* of blocks
(`[{type:"text"|"reasoning",...}]`) where the spec requires a string. No `compat`
flag covers this, and the symptom is subtle: short replies look fine, but any
response with reasoning blocks renders as `[object Object][object Object]`
before the text. `~/.prime/agent/extensions/databricks.ts` flattens each delta.

**Proxy.** Bun's fetch takes `init.proxy`; Node's undici does not, so the same
routing is an undici `ProxyAgent` dispatcher. Same file.

That extension is scoped to Databricks URLs only and delegates everything else
untouched. It exists solely because Prime has no per-provider fetch hook
(`StreamOptions` carries no `fetch`).

## Credentials

Secrets live only in `~/.prime/agent/auth.json` (mode `0600`); `models.json`
carries no key material. Prime's resolution order puts `auth.json` ahead of
environment variables and ahead of `models.json`, so the `apiKey` fields there
are env-var *names* acting as fallbacks.

```json
{
  "copilot":    { "type": "api_key", "key": "gho_..." },
  "databricks": { "type": "api_key", "key": "dapi..." }
}
```

## Verifying

Test the path you actually use. Tools change the request shape, and daemon
workers are a different code path from print mode:

```bash
# with tools (catches the strict-mode rejection)
prime-agent -p --model databricks/databricks-claude-opus-5 "Reply: OK"

# reasoning output (catches the array-content bug)
prime-agent -p --model databricks/databricks-claude-opus-5 \
  "Think carefully then answer: what is 17*23?"

# the full stack, through the kernel's MCP client
prime-agent -p --model copilot/claude-sonnet-5 \
  'Get the Noesis context plane and report experts/skills counts'
```

## Known non-issues

- **429 `REQUEST_LIMIT_EXCEEDED`** from Databricks is a workspace rate limit on
  input tokens per minute, not configuration. Prime's auto-retry usually rides
  through it; a higher FMAPI tier is an account matter.
- **Adherence.** Given the exact call, models return the right answer through
  the kernel. Left to themselves they sometimes probe with `pip`/`env` or read
  `knowledge_bases/*.py` instead of querying the graph -- which is right by luck
  for seed data and wrong for anything live. `AGENTS.md` states the rule
  emphatically; it is not yet reliably followed.
