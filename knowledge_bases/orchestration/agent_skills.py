"""Neo4j persistence for harness `AgentSkill` nodes.

Separate from `knowledge_bases.tools.repository` on purpose: that one stores
callable `Tool` definitions (handler + input schema), this one stores the
instruction bundles a harness discloses to a model. They were the same word
once, which is exactly the confusion this split removes.

Why the body lives here rather than in a file: a harness reads a skill's body
*on demand*, at the moment the model decides the skill is relevant. That read
can be served from the graph, so refining an instruction takes effect on the
next lookup -- no re-export, no restart, and every session shares one current
copy.
"""
from __future__ import annotations

from typing import Any

from graph import Graph
from knowledge_bases.orchestration.models import AgentSkill


class AgentSkillRepository:
    def __init__(self, graph: Graph):
        self.graph = graph

    def sync_skill(self, skill: AgentSkill) -> None:
        """MERGE one instruction bundle, bumping `version` when the body or
        wording actually changed so a refinement is visible as a new version
        rather than a silent overwrite."""
        skill.validate_for_harness()
        self.graph.run(
            "MERGE (s:AgentSkill {key: $key}) "
            "ON CREATE SET s.version = 1, s.created_at = datetime() "
            "SET s.version = CASE WHEN s.description <> $description "
            "                       OR s.body <> $body OR s.name <> $name "
            "                  THEN coalesce(s.version, 1) + 1 "
            "                  ELSE coalesce(s.version, 1) END, "
            "s.name = $name, s.description = $description, s.body = $body, "
            "s.knowledge_base = $knowledge_base, s.active = $active, "
            "s.updated_at = datetime()",
            key=skill.key, name=skill.name, description=skill.description,
            body=skill.body, knowledge_base=skill.knowledge_base,
            active=skill.active,
        )

    @staticmethod
    def _to_skill(row: dict[str, Any]) -> AgentSkill:
        data = {k: v for k, v in row.items() if v is not None}
        data.setdefault("body", "")
        return AgentSkill.model_validate(data)

    def load_skills(self, include_inactive: bool = False) -> list[AgentSkill]:
        """Every skill a harness should currently see, ordered for stability."""
        guard = "" if include_inactive else "WHERE s.active = true "
        rows = self.graph.run(
            f"MATCH (s:AgentSkill) {guard}"
            "RETURN s.key AS key, s.name AS name, s.description AS description, "
            "s.body AS body, s.knowledge_base AS knowledge_base, "
            "s.version AS version, s.active AS active ORDER BY s.key"
        )
        return [self._to_skill(row) for row in rows]

    def get_skill(self, key: str) -> AgentSkill | None:
        """One skill by key -- the live lookup a harness makes when the model
        opens a skill whose stub told it to ask the graph."""
        rows = self.graph.run(
            "MATCH (s:AgentSkill {key: $key}) "
            "RETURN s.key AS key, s.name AS name, s.description AS description, "
            "s.body AS body, s.knowledge_base AS knowledge_base, "
            "s.version AS version, s.active AS active",
            key=key,
        )
        return self._to_skill(rows[0]) if rows else None

    def set_active(self, key: str, active: bool) -> dict[str, Any]:
        """Retire or restore one skill without deleting it or its history."""
        rows = self.graph.run(
            "MATCH (s:AgentSkill {key: $key}) "
            "WITH s, s.active AS previous "
            "SET s.active = $active, s.updated_at = datetime() "
            "RETURN previous",
            key=key, active=active,
        )
        if not rows:
            raise ValueError(f"Unknown agent skill {key!r}")
        return {"key": key, "active": active, "previously_active": rows[0]["previous"]}
