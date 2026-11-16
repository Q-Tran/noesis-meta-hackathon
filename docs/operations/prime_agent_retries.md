# Riding out Databricks 429s

Databricks endpoints rate-limit hard. On the stock Prime Agent defaults a
rate-limited turn dies after ~14 seconds and waits for a human to type
"continue", which on a long autonomous run means the run is effectively
babysat.

This is a Prime Agent configuration, not Noesis code. It lives here because
the failure shows up while driving Noesis, and the reasoning is worth keeping.

## What the defaults do

`~/.prime/agent/settings.json`, unset, resolves to:

| Setting | Default | Effect |
|---|---|---|
| `retry.maxRetries` | `3` | 3 attempts, then give up |
| `retry.baseDelayMs` | `2000` | 2s, 4s, 8s (exponential) |
| `retry.provider.maxRetryDelayMs` | `60000` | **abort if the server asks to wait >60s** |

Two separate problems:

1. **The window is far too short.** 2s + 4s + 8s = **14 seconds** of total
   patience. A provider-side rate limit routinely lasts minutes, so the retries
   are exhausted long before the quota window rolls over. They were never going
   to succeed; they just fail faster.

2. **`maxRetryDelayMs` turns a long wait into an immediate abort.** This is the
   more damaging one, and it is counter-intuitive. When a 429 carries
   `Retry-After: 120s`, Prime does *not* wait 120s and retry -- it treats the
   requested delay as exceeding the cap and gives up **instantly**, without
   retrying at all:

   ```
   Retry-After 90s   -> {"kind":"exceeds-cap"}   # no retry
   Retry-After 120s  -> {"kind":"exceeds-cap"}   # no retry
   ```

   So the clearer the server is about how long to wait, the sooner Prime quits.
   The cap is a sane guard against a provider parking you for an hour, but at
   60s it is below the wait a busy Databricks endpoint actually asks for.

## The configuration

```json
"retry": {
  "enabled": true,
  "maxRetries": 8,
  "baseDelayMs": 2000,
  "provider": { "maxRetryDelayMs": 0 }
}
```

- **`maxRetries: 8`** -- the backoff is exponential, so this buys
  2+4+8+16+32+64+128+256 = **510s (8.5 min)** of patience, up from 14s. Because
  each attempt doubles, most of that budget is spent in the last two waits: if
  the limit clears early the run resumes early, and nothing is lost by having
  the tail available.
- **`maxRetryDelayMs: 0`** -- disables the cap, so a server-requested
  `Retry-After` is honored as given. Honoring what the provider explicitly asks
  for is strictly better information than any local guess; capping it means
  ignoring the one authoritative number in the exchange.

Retries are automatic: the session calls `agent.continue()` itself after
sleeping, so no human "continue" is needed.

## Verified

Against Prime's own modules, not by assumption:

```
retry settings : {"enabled":true,"maxRetries":8,"baseDelayMs":2000}
provider retry : {"maxRetryDelayMs":0}

no Retry-After   -> {"kind":"wait","delayMs":2000}
Retry-After 30s  -> {"kind":"wait","delayMs":30000}
Retry-After 120s -> {"kind":"wait","delayMs":120000}   # was: exceeds-cap
Retry-After 600s -> {"kind":"wait","delayMs":600000}   # was: exceeds-cap
```

## What this does not fix

- **Non-429 failures.** `invalid_request`, `refusal`, and `permission` are
  classified permanent and never retried, which is correct -- retrying sends
  the identical payload.
- **A genuinely exhausted quota.** If the limit lasts longer than ~8.5 minutes
  and the server sends no `Retry-After`, the turn still fails. Raise
  `maxRetries` further if that happens; each +1 roughly doubles the window.
- **Ctrl-C still works.** `abortRetry()` cancels a pending sleep, so a long
  wait is interruptible.

## Reverting

The previous file is backed up alongside it as
`settings.json.bak-<timestamp>`. Deleting the `retry` block restores the
defaults.
