# Catalog renames: cleaning up a pre-rename graph

`sync_tools_catalog.py` only `MERGE`s — it never deletes or retires. When a
knowledge base is renamed in code, the next sync writes the *new* `Tool`,
`KnowledgeBaseExpert`, and `AgentSkill` keys, but the old ones stay
`active = true` in the graph: the old expert keeps answering routed questions
and the old tools keep being listed (their `handler` strings point at modules
that no longer exist, so calling them fails).

After syncing a renamed/refolded catalog, retire the stale keys rather than
deleting them — spans, routing traces, and overrides still reference them:

```cypher
// Stale tools from renamed or removed knowledge bases
MATCH (t:Tool)
WHERE t.knowledge_base IN ["evidence", "decision_intelligence", "delivery",
                           "snowflake", "bmad", "execution"]
SET t.active = false, t.updated_at = datetime();

// Stale experts under the old keys
MATCH (e:KnowledgeBaseExpert)
WHERE e.key IN ["evidence", "decision_intelligence", "delivery",
                "snowflake", "bmad", "execution"]
SET e.active = false, e.updated_at = datetime();

// Stale instruction bundles
MATCH (s:AgentSkill)
WHERE s.key IN ["noesis-evidence", "noesis-decision-intelligence",
                "noesis-delivery", "noesis-snowflake",
                "noesis-bmad", "noesis-execution"]
SET s.active = false, s.updated_at = datetime();
```

Order matters: run `sync_tools_catalog.py` *first* (it writes the new keys and
the `AgentSkill -[:DOCUMENTS]-> Tool` edges), then the retirement statements,
then `export_prime_context.py` (or let the sync's auto-export do it — it also
deletes the stale `.prime/agent/skills/noesis-*/` stub directories).

Because retirement is just `active = false`, everything is reversible: a
mistakenly retired key comes back by setting `active = true`, and a re-sync of
the old code restores it outright.
