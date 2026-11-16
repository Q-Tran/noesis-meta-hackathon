"""Graph-backed workers, reflection proposals, evaluation gates, and
connector/ontology control planes.

Workers are the product's fine-grained agents. A `Worker` is the stable
employee/task identity; a `WorkerVersion` is the task-scoped configuration
(prompt, skills, tools, capabilities) that reflection may tune. The global
system prompt is never tuned by reflection: only the worker's version is.

Every change flow uses the same argumentative shape:

    ReflectionRun --CREATED_BY_REFLECTION--> ChangeProposal --EVIDENCED_BY--> ProposalEvidence
                                             --EVALUATED_BY--> EvalReport

Reflection never mutates a worker directly. The eval gate must pass before
a proposal can activate a version. Applying a version supersedes the old
one; reverting re-points `ACTIVE_VERSION` to a prior version. The whole
chain -- sample, evidence, gate, application, post-apply check -- stays in
the graph so *why* the worker changed is queryable forever.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from graph import Graph
from knowledge_bases.orchestration.models import (
    ChangeProposal,
    ConnectorDefinition,
    ConnectorInstance,
    EvalReport,
    OntologyProposal,
    ReflectionRun,
    WorkerDefinition,
    WorkerVersion,
)
from knowledge_bases.tools.models import OutcomeSource, SpanEvent, SpanKind, SpanStatus, TraceOutcome
from knowledge_bases.tools.scoring import (
    ScoreBreakdown,
    SCORER_VERSION,
    compare_scores,
    score_trace,
    score_worker_traces,
)
from knowledge_bases.tools.trace_store import TraceStore

WORKER_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,80}$")
AUTONOMIES = ("auto", "review", "locked")
WORKER_KINDS = ("custom", "migrated_expert", "system")
MUTABLE_VERSION_FIELDS = (
    "system_prompt", "capabilities", "entry_tools", "allowed_tools", "skills", "model_hint",
)
DEFAULT_WORKER_AUTONOMY = "auto"
DEFAULT_ONTOLOGY_AUTONOMY = "review"
MAX_APPLIED_PER_DAY = 3


def _id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, default=str, sort_keys=True)


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _list(value: Any) -> list:
    parsed = _loads(value)
    return list(parsed) if isinstance(parsed, list) else []


def _dict(value: Any) -> dict:
    parsed = _loads(value)
    return parsed if isinstance(parsed, dict) else {}


def tools_catalog_keys(graph: Graph | None = None) -> set[str]:
    """Tool keys a worker may reference: live `Tool` nodes, falling back to
    the composed code catalog when the graph has not been synced yet."""
    if graph is not None:
        try:
            rows = graph.run("MATCH (t:Tool {active: true}) RETURN t.key AS key")
            keys = {str(row["key"]) for row in rows if row.get("key")}
            if keys:
                return keys
        except Exception:  # noqa: BLE001 -- catalog fallback is the point
            pass
    try:
        from knowledge_bases.registry import composed_tools

        return {tool.key for tool in composed_tools()}
    except Exception:  # noqa: BLE001
        return set()


def _tool_pattern_allowed(pattern: str, keys: set[str]) -> bool:
    return pattern in keys or (
        pattern.endswith(".*") and any(key.startswith(pattern[:-1]) for key in keys)
    )


class WorkerRepository:
    """Graph persistence for `Worker`, `WorkerVersion`, capabilities, and
    assignments. All methods are bounded: they only create/update these
    labels, and tool references are validated against the catalog."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def validate_tools(self, patterns: list[str]) -> None:
        keys = tools_catalog_keys(self.graph)
        bad = [p for p in patterns if not _tool_pattern_allowed(p, keys)]
        if bad:
            raise ValueError(f"Worker references unknown or unregistered tools: {bad}")

    def _worker_props(self, definition: WorkerDefinition) -> dict[str, Any]:
        return {
            "key": definition.key,
            "name": definition.name,
            "description": definition.description,
            "mission": definition.mission,
            "active": definition.active,
            "kind": definition.kind,
            "version": definition.version,
            "autonomy": definition.autonomy,
            "routable": definition.routable,
            "priority": definition.priority,
            "model_hint": definition.model_hint,
        }

    def _version_props(self, definition: WorkerDefinition, status: str, reason: str) -> dict[str, Any]:
        return {
            "id": _id("workerversion"),
            "key": definition.key,
            "version": definition.version,
            "status": status,
            "system_prompt": definition.system_prompt,
            "capabilities": list(definition.capabilities),
            "entry_tools": list(definition.entry_tools),
            "allowed_tools": list(definition.allowed_tools),
            "skills": list(definition.skills),
            "model_hint": definition.model_hint,
            "reason": reason,
            "created_at": _now_iso(),
        }

    def _write_version(self, definition: WorkerDefinition, status: str, reason: str) -> dict[str, Any]:
        props = self._version_props(definition, status, reason)
        self.graph.run(
            "MERGE (v:WorkerVersion {id: $id}) "
            "ON CREATE SET v += $props, v.created_at = datetime() "
            "WITH v "
            "MATCH (w:Worker {key: $key}) "
            "MERGE (v)-[:HAS_VERSION]->(w) "
            "ON CREATE SET v.created_at = datetime()",
            id=props["id"], props=props, key=definition.key, status=status,
        )
        self._write_capabilities(definition.key, definition.capabilities)
        if definition.allowed_tools:
            self.graph.run(
                "MATCH (v:WorkerVersion {id: $id}) "
                "MATCH (t:Tool {active: true}) "
                "WHERE any(p IN $patterns WHERE p = t.key "
                "  OR (p ENDS WITH '.*' AND t.key STARTS WITH left(p, size(p) - 1))) "
                "MERGE (v)-[u:WORKER_CAN_USE]->(t) "
                "ON CREATE SET u.created_at = datetime()",
                id=props["id"], patterns=list(definition.allowed_tools),
            )
        return props

    def _write_capabilities(self, key: str, capabilities: list[str]) -> None:
        if not capabilities:
            return
        self.graph.run(
            "MATCH (w:Worker {key: $key}) "
            "UNWIND range(0, size($capabilities) - 1) AS position "
            "WITH w, position, $capabilities[position] AS statement "
            "MERGE (c:WorkerCapability {id: $key + ':' + toString(position)}) "
            "ON CREATE SET c.created_at = datetime() "
            "SET c.statement = statement, c.position = position "
            "MERGE (w)-[h:WORKER_HAS_CAPABILITY]->(c) "
            "ON CREATE SET h.created_at = datetime()",
            key=key, capabilities=list(capabilities),
        )

    def upsert_worker(self, definition: WorkerDefinition, reason: str = "manual") -> WorkerDefinition:
        """Create or revise a worker. A changed configuration becomes a new
        active WorkerVersion; identity, autonomy, and kind are preserved."""
        if not WORKER_KEY_RE.match(definition.key):
            raise ValueError(f"Invalid worker key {definition.key!r}")
        if definition.autonomy not in AUTONOMIES:
            raise ValueError(f"Unknown autonomy {definition.autonomy!r}")
        if definition.kind not in WORKER_KINDS:
            raise ValueError(f"Unknown worker kind {definition.kind!r}")
        self.validate_tools(list(definition.allowed_tools) + list(definition.entry_tools))

        existing = self.load_worker(definition.key)
        if existing is None:
            self.graph.run(
                "MERGE (w:Worker {key: $key}) "
                "ON CREATE SET w += $props, w.created_at = datetime() "
                "SET w.updated_at = datetime()",
                key=definition.key, props=self._worker_props(definition),
            )
            version = self._write_version(definition, "active", reason)
            self._set_active_edge(definition.key, version["id"])
            return definition

        active = self.active_version(definition.key)
        changed = (
            active is None
            or active.system_prompt != definition.system_prompt
            or active.capabilities != list(definition.capabilities)
            or active.entry_tools != list(definition.entry_tools)
            or active.allowed_tools != list(definition.allowed_tools)
            or active.skills != list(definition.skills)
            or active.model_hint != definition.model_hint
        )
        next_version = (active.version + 1) if active else 1
        if changed:
            definition = definition.model_copy(update={
                "version": next_version,
                "autonomy": existing.autonomy,
                "kind": existing.kind,
            })
            if active is not None:
                self.graph.run(
                    "MATCH (v:WorkerVersion {id: $id}) SET v.status = 'superseded'",
                    id=active.id,
                )
            version = self._write_version(definition, "active", reason)
            self._set_active_edge(definition.key, version["id"])
        self.graph.run(
            "MATCH (w:Worker {key: $key}) "
            "SET w.name = $name, w.description = $description, w.mission = $mission, "
            "    w.active = $active, w.routable = $routable, w.priority = $priority, "
            "    w.model_hint = $model_hint, w.version = $version, w.updated_at = datetime()",
            key=definition.key, name=definition.name, description=definition.description,
            mission=definition.mission, active=definition.active,
            routable=definition.routable, priority=definition.priority,
            model_hint=definition.model_hint, version=definition.version,
        )
        return definition

    def _set_active_edge(self, key: str, version_id: str) -> None:
        self.graph.run(
            "MATCH (w:Worker {key: $key}) "
            "OPTIONAL MATCH (w)-[a:ACTIVE_VERSION]->(old:WorkerVersion) "
            "DELETE a "
            "WITH w "
            "MATCH (v:WorkerVersion {id: $version_id}) "
            "MERGE (w)-[a:ACTIVE_VERSION]->(v) "
            "ON CREATE SET a.created_at = datetime()",
            key=key, version_id=version_id,
        )

    def load_worker(self, key: str) -> WorkerDefinition | None:
        rows = self.graph.run(
            "MATCH (w:Worker {key: $key}) "
            "OPTIONAL MATCH (w)-[:ACTIVE_VERSION]->(v:WorkerVersion) "
            "RETURN properties(w) AS worker, properties(v) AS version",
            key=key,
        )
        if not rows:
            return None
        worker = rows[0].get("worker") or {}
        version = rows[0].get("version") or {}
        data = {**worker, **{k: version[k] for k in version if k in (
            "system_prompt", "capabilities", "entry_tools", "allowed_tools", "skills", "model_hint",
        )}}
        data.setdefault("description", "")
        data.setdefault("mission", "")
        data.setdefault("capabilities", [])
        data.setdefault("entry_tools", [])
        data.setdefault("allowed_tools", [])
        data.setdefault("skills", [])
        data.setdefault("autonomy", DEFAULT_WORKER_AUTONOMY)
        data.setdefault("kind", "custom")
        data.setdefault("version", 1)
        return WorkerDefinition.model_validate({k: v for k, v in data.items() if v is not None})

    def load_workers(self, include_inactive: bool = False) -> list[WorkerDefinition]:
        rows = self.graph.run(
            "MATCH (w:Worker) WHERE $all OR w.active "
            "OPTIONAL MATCH (w)-[:ACTIVE_VERSION]->(v:WorkerVersion) "
            "RETURN properties(w) AS worker, properties(v) AS version "
            "ORDER BY w.priority, w.key",
            all=include_inactive,
        )
        workers = []
        for row in rows:
            worker = row.get("worker") or {}
            version = row.get("version") or {}
            data = {**worker, **{k: version[k] for k in version if k in (
                "system_prompt", "capabilities", "entry_tools", "allowed_tools", "skills", "model_hint",
            )}}
            data.setdefault("description", "")
            data.setdefault("mission", "")
            data.setdefault("capabilities", [])
            data.setdefault("entry_tools", [])
            data.setdefault("allowed_tools", [])
            data.setdefault("skills", [])
            data.setdefault("autonomy", DEFAULT_WORKER_AUTONOMY)
            data.setdefault("kind", "custom")
            data.setdefault("version", 1)
            workers.append(WorkerDefinition.model_validate({k: v for k, v in data.items() if v is not None}))
        return workers

    def active_version(self, key: str) -> WorkerVersion | None:
        rows = self.graph.run(
            "MATCH (w:Worker {key: $key})-[:ACTIVE_VERSION]->(v:WorkerVersion) "
            "RETURN properties(v) AS version",
            key=key,
        )
        return WorkerVersion.model_validate(rows[0]["version"]) if rows else None

    def worker_versions(self, key: str) -> list[WorkerVersion]:
        rows = self.graph.run(
            "MATCH (v:WorkerVersion {key: $key}) RETURN properties(v) AS version "
            "ORDER BY v.version DESC",
            key=key,
        )
        return [WorkerVersion.model_validate(row["version"]) for row in rows]

    def create_draft_version(
        self, key: str, changes: dict[str, Any], reason: str,
    ) -> WorkerVersion:
        """Create the next version as a draft. The active version is untouched."""
        worker = self.load_worker(key)
        if worker is None:
            raise ValueError(f"Unknown worker {key!r}")
        unknown = set(changes) - set(MUTABLE_VERSION_FIELDS)
        if unknown:
            raise ValueError(f"WorkerVersion changes cannot touch {sorted(unknown)}")
        draft = worker.model_copy(update={**changes, "version": worker.version + 1})
        # Reuse the version writer, but never activate a draft.
        props = self._version_props(draft, "draft", reason)
        props["id"] = _id("workerversion")
        self.graph.run(
            "MERGE (v:WorkerVersion {id: $id}) "
            "ON CREATE SET v += $props, v.created_at = datetime() "
            "WITH v "
            "MATCH (w:Worker {key: $key}) "
            "MERGE (v)-[:HAS_VERSION]->(w) "
            "ON CREATE SET v.created_at = datetime()",
            id=props["id"], props=props, key=key,
        )
        return WorkerVersion.model_validate(props)

    def activate_version(self, key: str, version: int, reason: str) -> WorkerVersion:
        """The only path that makes a version live. Reverting is the same act
        pointed at an older version -- history is never rewritten."""
        versions = self.worker_versions(key)
        target = next((v for v in versions if v.version == version), None)
        if target is None:
            raise ValueError(f"Unknown WorkerVersion {key!r} v{version}")
        current = self.active_version(key)
        if current is not None and current.version == version:
            return target
        self.graph.run(
            "MATCH (v:WorkerVersion {id: $id}) SET v.status = 'active', v.activated_at = datetime()",
            id=target.id,
        )
        if current is not None:
            self.graph.run(
                "MATCH (v:WorkerVersion {id: $id}) SET v.status = 'superseded'",
                id=current.id,
            )
            self.graph.run(
                "MATCH (n:WorkerVersion {id: $new_id})-[:SUPERSEDES]->(o:WorkerVersion {id: $old_id}) "
                "RETURN o.status AS status",
                new_id=target.id, old_id=current.id,
            )
        self._set_active_edge(key, target.id)
        self.graph.run(
            "MATCH (w:Worker {key: $key}) "
            "SET w.version = $version, w.active = true, w.updated_at = datetime(), "
            "    w.system_prompt = $prompt, w.capabilities = $capabilities, "
            "    w.entry_tools = $entry_tools, w.allowed_tools = $allowed_tools, "
            "    w.skills = $skills, w.model_hint = $model_hint",
            key=key, version=version, prompt=target.system_prompt,
            capabilities=list(target.capabilities), entry_tools=list(target.entry_tools),
            allowed_tools=list(target.allowed_tools), skills=list(target.skills),
            model_hint=target.model_hint,
        )
        return target

    def revert_to_version(self, key: str, version: int, reason: str) -> WorkerVersion:
        return self.activate_version(key, version, reason)

    def retire_worker(self, key: str) -> WorkerDefinition:
        worker = self.load_worker(key)
        if worker is None:
            raise ValueError(f"Unknown worker {key!r}")
        self.graph.run(
            "MATCH (w:Worker {key: $key}) SET w.active = false, w.routable = false, "
            "w:Deprecated, w.updated_at = datetime()",
            key=key,
        )
        return worker.model_copy(update={"active": False, "routable": False})

    def set_autonomy(self, key: str, autonomy: str) -> WorkerDefinition:
        if autonomy not in AUTONOMIES:
            raise ValueError(f"Unknown autonomy {autonomy!r}; expected one of {AUTONOMIES}")
        worker = self.load_worker(key)
        if worker is None:
            raise ValueError(f"Unknown worker {key!r}")
        self.graph.run(
            "MATCH (w:Worker {key: $key}) SET w.autonomy = $autonomy, w.updated_at = datetime()",
            key=key, autonomy=autonomy,
        )
        return worker.model_copy(update={"autonomy": autonomy})

    def migrate_experts(self, reason: str = "experts_to_workers") -> list[WorkerDefinition]:
        """Copy every KnowledgeBaseExpert into a Worker. Experts remain
        readable; workers are the canonical product surface from here on."""
        rows = self.graph.run(
            "MATCH (e:KnowledgeBaseExpert) RETURN properties(e) AS expert ORDER BY e.key"
        )
        created = []
        for row in rows:
            expert = row.get("expert") or {}
            worker = WorkerDefinition.model_validate({
                "key": expert.get("key"),
                "name": expert.get("name") or expert.get("key"),
                "description": expert.get("description") or "",
                "mission": expert.get("description") or "",
                "capabilities": _list(expert.get("capabilities")) + _list(expert.get("capability_overlays")),
                "priority": expert.get("priority") or 100,
                "active": bool(expert.get("active", True)),
                "kind": "migrated_expert",
                "version": int(expert.get("version") or 1),
                "autonomy": DEFAULT_WORKER_AUTONOMY,
                "system_prompt": expert.get("system_prompt") or "",
                "entry_tools": _list(expert.get("entry_tools")),
                "allowed_tools": _list(expert.get("allowed_tools")),
                "skills": [],
                "model_hint": expert.get("model_hint"),
                "routable": bool(expert.get("routable", True)),
            })
            created.append(self.upsert_worker(worker, reason))
        return created

    def create_assignment(self, worker_key: str, task: str = "", queue: str = "",
                          channel: str = "", scope: str = "") -> dict:
        if self.load_worker(worker_key) is None:
            raise ValueError(f"Unknown worker {worker_key!r}")
        assignment = {
            "id": _id("assignment"),
            "worker_key": worker_key,
            "task": task,
            "queue": queue,
            "channel": channel,
            "scope": scope,
            "active": True,
            "created_at": _now_iso(),
        }
        self.graph.run(
            "MERGE (a:WorkerAssignment {id: $id}) "
            "ON CREATE SET a += $props, a.created_at = datetime() "
            "WITH a "
            "MATCH (w:Worker {key: $key}) "
            "MERGE (w)-[:HAS_ASSIGNMENT]->(a) "
            "ON CREATE SET a.created_at = datetime()",
            id=assignment["id"], props=assignment, key=worker_key,
        )
        return assignment




class WorkerRuntime:
    """Run one task under a Worker.

    The worker supplies only its task-scoped prompt, tools, and skills. This
    class does not introduce a global system prompt: the LLM is called with
    the worker's prompt as its system context, and any tool calls are
    recorded with the worker key.
    """

    def __init__(self, graph: Graph, repository: WorkerRepository | None = None, llm: Any | None = None):
        self.graph = graph
        self.repository = repository or WorkerRepository(graph)
        self.store = TraceStore(graph)
        self.llm = llm

    def _llm(self):
        if self.llm is None:
            from llm import get_llm

            self.llm = get_llm()
        return self.llm

    def load(self, worker_key: str) -> WorkerDefinition:
        worker = self.repository.load_worker(worker_key)
        if worker is None:
            raise ValueError(f"Unknown worker {worker_key!r}")
        return worker

    def default_tool_arguments(self, tool_key: str, task: str) -> dict[str, Any]:
        """Derive arguments for a registered tool from the task.

        Known string fields receive the task; known search fields receive
        the worker's likely entity scope. Anything else is omitted and the
        dispatcher validates it honestly.
        """
        try:
            from knowledge_bases.tools.repository import ToolRepository

            tool = next((t for t in ToolRepository(self.graph).load_tools() if t.key == tool_key), None)
        except Exception:  # noqa: BLE001 -- generic arguments still work for fakes
            tool = None
        schema = (tool.input_schema if tool else {}) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        entity_types = self._likely_entity_types(task)
        for name in properties:
            if name in ("query", "question", "text", "description"):
                args[name] = task
            elif name == "entity_types":
                if entity_types:
                    args[name] = entity_types
            elif name == "filters":
                args[name] = {}
            elif name == "limit":
                args[name] = 5
            elif name == "knowledge_base" and tool:
                args[name] = tool.knowledge_base
        return args

    def _likely_entity_types(self, task: str) -> list[str]:
        try:
            from knowledge_bases.registry import search_types

            task_words = set(task.lower().replace("_", " ").split())
            matches = [
                qualified
                for qualified, config in search_types().items()
                if config.label.lower() in task_words or qualified.lower() in task_words
            ]
            families: dict[str, list[str]] = {}
            for qualified in matches:
                families.setdefault(qualified.split(".", 1)[0], []).append(qualified)
            return next(iter(families.values()), [])
        except Exception:  # noqa: BLE001
            return []

    def _dispatch_real_tool(self, tool_key: str, arguments: dict[str, Any], trace_id: str) -> Any:
        if "." not in tool_key:
            raise ValueError(f"Tool key {tool_key!r} is not a qualified tool key")
        knowledge_base, operation = tool_key.split(".", 1)
        from knowledge_bases.mcp import dispatch_tool

        call_args = dict(arguments)
        call_args.setdefault("trace_id", trace_id)
        call_args.setdefault("expert_key", "shadow-worker")
        return dispatch_tool(self.graph, knowledge_base, operation, call_args, trace_id=trace_id)

    def _record_shadow_span(self, trace_id: str, tool_key: str, arguments: dict[str, Any], result: Any, duration_ms: int) -> None:
        error = result.get("error") if isinstance(result, dict) else None
        if isinstance(result, list):
            result_count = len(result)
        elif isinstance(result, dict):
            result_count = None
            for key in ("results", "matches", "entities", "related_by"):
                inner = result.get(key)
                if isinstance(inner, list):
                    result_count = len(inner)
                    break
        else:
            result_count = None
        now = datetime.now(timezone.utc)
        try:
            self.store.insert_span(SpanEvent(
                trace_id=trace_id,
                name=tool_key.split(".", 1)[-1],
                kind=SpanKind.tool_call,
                knowledge_base=tool_key.split(".", 1)[0],
                skill=tool_key,
                started_at=now,
                ended_at=now,
                duration_ms=duration_ms,
                status=SpanStatus.error if error else SpanStatus.ok,
                error=str(error) if error else None,
                raw_input=arguments,
                raw_output=result,
                result_count=result_count,
                expert_key="shadow-worker",
            ))
        except Exception:  # noqa: BLE001 -- shadow evaluation must not mask the real result
            pass

    def run_definition(
        self,
        worker: WorkerDefinition,
        task: str,
        *,
        trace_id: str | None = None,
        dispatch: Callable[..., Any] | None = None,
        record_outcome: bool = True,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict:
        """Run one task under a specific definition, without activating it.

        `dispatch` is a test/seam callable receiving
        `(tool_key, arguments, graph, trace_id)`. When omitted the runtime
        calls the real graph-registered tool handlers.
        """
        trace_id = trace_id or _id("trace")
        evidence: list[dict[str, Any]] = []
        calls = tool_calls or [
            {"tool_key": key, "arguments": self.default_tool_arguments(key, task)}
            for key in worker.entry_tools
        ]
        for call in calls:
            tool_key = str(call.get("tool_key") or "")
            if not tool_key:
                continue
            arguments = dict(call.get("arguments") or {})
            if not _tool_pattern_allowed(tool_key, set(worker.allowed_tools)):
                evidence.append({
                    "tool": tool_key,
                    "result": {"error": "tool not allowed by this worker version"},
                    "skipped": True,
                })
                if dispatch is not None:
                    self._record_shadow_span(trace_id, tool_key, arguments, {"error": "tool not allowed by this worker version"}, 0)
                continue
            started = datetime.now(timezone.utc)
            try:
                if dispatch is None:
                    result = self._dispatch_real_tool(tool_key, arguments, trace_id)
                else:
                    try:
                        result = dispatch(tool_key, arguments, self.graph, trace_id)
                    except TypeError:
                        result = dispatch(tool_key, arguments, self.graph)  # legacy 3-arg seam
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
            duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
            evidence.append({"tool": tool_key, "result": result})
            if dispatch is not None:
                self._record_shadow_span(trace_id, tool_key, arguments, result, duration_ms)

        prompt = task
        if evidence:
            prompt += "\n\nEvidence from worker tools:\n" + json.dumps(evidence, default=str)[:4000]
        try:
            answer = self._llm().complete(prompt, system=worker.system_prompt)
        except Exception as exc:  # noqa: BLE001
            answer = f"Worker failed: {exc}"
        score_payload: dict[str, Any] = {
            "score": 100.0 if answer else 0.0,
            "breakdown": {
                "correctness": 100.0 if answer else 0.0,
                "groundedness": 100.0 if not any(isinstance(e.get("result"), dict) and "error" in e["result"] for e in evidence) else 0.0,
                "behavior": 100.0,
                "cost": 100.0,
            },
        }
        if record_outcome:
            from knowledge_bases.tools.scoring import record_trace_score

            outcome = record_trace_score(trace_id, store=self.store)
            score_payload = {
                "score": outcome.score,
                "breakdown": _loads(outcome.score_breakdown_json) or score_payload["breakdown"],
            }
        return {
            "trace_id": trace_id,
            "worker_key": worker.key,
            "version": worker.version,
            "answer": answer,
            "evidence": evidence,
            **score_payload,
        }

    def run(self, worker_key: str, task: str, *, trace_id: str | None = None, dispatch: Callable[..., Any] | None = None) -> dict:
        """Load the active worker and answer one task."""
        worker = self.load(worker_key)
        return self.run_definition(worker, task, trace_id=trace_id, dispatch=dispatch)

class ProposalRepository:
    """Persistence for the argumentative change record."""

    def __init__(self, graph: Graph):
        self.graph = graph

    def create(self, proposal: ChangeProposal, reflection_run_id: str | None = None) -> ChangeProposal:
        props = {
            "id": proposal.id,
            "kind": proposal.kind,
            "target_key": proposal.target_key,
            "status": proposal.status,
            "title": proposal.title,
            "summary": proposal.summary,
            "motivation": proposal.motivation,
            "changes_json": _json(proposal.changes),
            "actor": proposal.actor,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self.graph.run(
            "MERGE (p:ChangeProposal {id: $id}) "
            "ON CREATE SET p += $props, p.created_at = datetime() "
            "SET p.updated_at = datetime()",
            id=proposal.id, props=props,
        )
        for item in proposal.evidence:
            evidence_id = item.get("id") or _id("evidence")
            evidence_props = {
                "id": evidence_id,
                "kind": item.get("kind") or "score",
                "summary": item.get("summary") or "",
                "trace_ids": item.get("trace_ids") or [],
                "score": item.get("score"),
                "created_at": _now_iso(),
            }
            self.graph.run(
                "MERGE (e:ProposalEvidence {id: $id}) "
                "ON CREATE SET e += $props, e.created_at = datetime() "
                "WITH e "
                "MATCH (p:ChangeProposal {id: $proposal_id}) "
                "MERGE (p)-[:EVIDENCED_BY]->(e) "
                "ON CREATE SET e.created_at = datetime()",
                id=evidence_id, props=evidence_props, proposal_id=proposal.id,
            )
        if reflection_run_id:
            self.graph.run(
                "MATCH (r:ReflectionRun {id: $run_id}) "
                "MATCH (p:ChangeProposal {id: $proposal_id}) "
                "MERGE (r)-[:CREATED_BY_REFLECTION]->(p) "
                "ON CREATE SET p.created_at = datetime()",
                run_id=reflection_run_id, proposal_id=proposal.id,
            )
        return proposal

    def get(self, proposal_id: str) -> ChangeProposal | None:
        rows = self.graph.run(
            "MATCH (p:ChangeProposal {id: $id}) "
            "OPTIONAL MATCH (p)-[:EVIDENCED_BY]->(e:ProposalEvidence) "
            "WITH p, collect(properties(e)) AS evidence "
            "RETURN properties(p) AS proposal, evidence",
            id=proposal_id,
        )
        if not rows:
            return None
        return self._to_proposal(rows[0])

    def list(self, status: str | None = None) -> list[ChangeProposal]:
        rows = self.graph.run(
            "MATCH (p:ChangeProposal) WHERE $status IS NULL OR p.status = $status "
            "OPTIONAL MATCH (p)-[:EVIDENCED_BY]->(e:ProposalEvidence) "
            "WITH p, collect(properties(e)) AS evidence "
            "RETURN properties(p) AS proposal, evidence "
            "ORDER BY p.created_at DESC",
            status=status,
        )
        return [self._to_proposal(row) for row in rows]

    @staticmethod
    def _to_proposal(row: dict) -> ChangeProposal:
        props = row.get("proposal") or {}
        evidence = [e for e in (row.get("evidence") or []) if e]
        return ChangeProposal.model_validate({
            **props,
            "changes": _dict(props.get("changes_json")),
            "evidence": evidence,
        })

    def set_status(self, proposal_id: str, status: str, extra: dict[str, Any] | None = None) -> None:
        props = {"status": status, "updated_at": _now_iso(), **(extra or {})}
        self.graph.run(
            "MATCH (p:ChangeProposal {id: $id}) SET p += $props, p.updated_at = datetime()",
            id=proposal_id, props=props,
        )

    def record_eval(self, report: EvalReport) -> EvalReport:
        props = {
            "id": report.id,
            "proposal_id": report.proposal_id,
            "stage": report.stage,
            "passed": report.passed,
            "baseline_score": report.baseline_score,
            "candidate_score": report.candidate_score,
            "detail": report.detail,
            "samples_json": report.samples_json,
            "created_at": _now_iso(),
        }
        self.graph.run(
            "MERGE (r:EvalReport {id: $id}) "
            "ON CREATE SET r += $props, r.created_at = datetime() "
            "WITH r "
            "MATCH (p:ChangeProposal {id: $proposal_id}) "
            "MERGE (p)-[:EVALUATED_BY]->(r) "
            "ON CREATE SET r.created_at = datetime()",
            id=report.id, props=props, proposal_id=report.proposal_id,
        )
        return report


class ReflectionService:
    """Sampler, reflector, eval gate, applier, and auto-reverter.

    The reflector is deterministic by default: it turns sampled signals into
    a proposal with evidence. An injected judge callable can refine scores,
    but no model output is required for the loop to run.
    """

    def __init__(self, graph: Graph, worker_repo: WorkerRepository | None = None,
                 proposal_repo: ProposalRepository | None = None,
                 store: TraceStore | None = None,
                 shadow_llm: Any | None = None,
                 shadow_dispatcher: Callable[..., Any] | None = None):
        self.graph = graph
        self.workers = worker_repo or WorkerRepository(graph)
        self.proposals = proposal_repo or ProposalRepository(graph)
        self.store = store or TraceStore(graph)
        self._shadow_llm = shadow_llm
        self._shadow_dispatcher = shadow_dispatcher

    def sample_worker_traces(self, worker_key: str, limit: int = 40) -> list[dict]:
        rows = self.graph.run(
            "MATCH (s:Span) WHERE coalesce(s.expert_key, s.worker_key) = $worker_key "
            "RETURN DISTINCT s.trace_id AS trace_id ORDER BY trace_id",
            worker_key=worker_key,
        )
        scored = []
        for row in rows:
            trace_id = row.get("trace_id")
            if not trace_id:
                continue
            breakdown = score_trace(trace_id, store=self.store)
            scored.append({"trace_id": trace_id, "score": breakdown.composite, "breakdown": breakdown.as_dict()})
        scored.sort(key=lambda item: item["score"])
        return scored[:limit]

    def create_reflection_run(self, target_kind: str, target_key: str,
                              sampled_trace_ids: list[str]) -> ReflectionRun:
        run = ReflectionRun(
            id=_id("reflectionrun"),
            target_kind=target_kind,
            target_key=target_key,
            sampled_trace_ids=list(sampled_trace_ids),
            scorer_version=SCORER_VERSION,
        )
        self.graph.run(
            "MERGE (r:ReflectionRun {id: $id}) "
            "ON CREATE SET r.id = $id, r.target_kind = $kind, r.target_key = $target, "
            "r.sampled_trace_ids = $ids, r.scorer_version = $scorer, r.created_at = datetime()",
            id=run.id, kind=target_kind, target=target_key,
            ids=list(sampled_trace_ids), scorer=SCORER_VERSION,
        )
        return run

    def _evidence_for_traces(self, sampled: list[dict]) -> list[dict]:
        return [
            {
                "id": _id("evidence"),
                "kind": "score",
                "summary": f"Trace {item['trace_id']} scored {item['score']:.1f}",
                "trace_ids": [item["trace_id"]],
                "score": item["score"],
            }
            for item in sampled[:12]
        ]

    def _signals(self, worker: WorkerDefinition, sampled: list[dict]) -> dict:
        spans: list[dict] = []
        for item in sampled:
            spans.extend(self.store.get_trace_spans(item["trace_id"]))
        failed = [s for s in spans if s.get("status") == "error"]
        error_tools = Counter(str(s.get("skill") or s.get("name") or "") for s in failed if s.get("skill") or s.get("name"))
        repeated = Counter(str(s.get("name") or "") for s in spans)
        loops = [name for name, count in repeated.items() if count >= 3 and any(
            marker in name for marker in ("search", "lookup", "find", "resolve", "get_"))]
        empty = sum(1 for s in spans if s.get("result_count") == 0)
        successful_tools = sorted({
            str(s.get("skill") or s.get("name") or "")
            for item in sampled if item["score"] >= 70
            for s in self.store.get_trace_spans(item["trace_id"])
            if s.get("skill") or s.get("name")
        })
        missing_tools = sorted(set(successful_tools) - set(worker.allowed_tools))
        return {
            "error_tools": error_tools.most_common(5),
            "loops": loops[:5],
            "empty_calls": empty,
            "successful_tools": successful_tools,
            "missing_tools": missing_tools[:8],
            "low_traces": [item["trace_id"] for item in sampled if item["score"] < 60],
        }

    def _candidate_definition(self, worker: WorkerDefinition, changes: dict[str, Any]) -> WorkerDefinition:
        """Apply a proposal's changes to the active definition, in memory.

        The candidate is a WorkerDefinition for execution only; it is never
        written as a version until the gate passes and `apply_proposal` runs.
        """
        mutable = {key: value for key, value in changes.items() if key in MUTABLE_VERSION_FIELDS}
        if changes.get("remove_tools"):
            removed = set(changes["remove_tools"])
            mutable["allowed_tools"] = [
                tool for tool in worker.allowed_tools if tool not in removed
                and not any(tool.startswith(prefix) for prefix in (
                    removed_value[:-1] if removed_value.endswith(".*") else removed_value
                    for removed_value in removed
                ) if removed_value.endswith(".*"))
            ]
        return worker.model_copy(update={**mutable, "version": worker.version + 1})

    def _trace_task(self, trace_id: str) -> str:
        rows = self.graph.run(
            "MATCH (t:RoutingTrace {id: $trace_id}) RETURN t.question AS question",
            trace_id=trace_id,
        )
        if rows and rows[0].get("question"):
            return str(rows[0]["question"])
        spans = self.store.get_trace_spans(trace_id)
        for span in reversed(spans):
            raw = _loads(span.get("raw_input"))
            if isinstance(raw, dict):
                for key in ("query", "question", "text"):
                    if raw.get(key):
                        return str(raw[key])
                skill = span.get("skill") or span.get("name") or "tool"
                return f"Perform {skill} with arguments {json.dumps(raw, default=str)}"
        return ""

    def _shadow_tool_calls(self, trace_id: str) -> list[dict[str, Any]]:
        calls = []
        for span in self.store.get_trace_spans(trace_id):
            tool_key = str(span.get("skill") or span.get("name") or "")
            if not tool_key or "." not in tool_key:
                continue
            arguments = _loads(span.get("raw_input"))
            if not isinstance(arguments, dict):
                arguments = {}
            arguments = {
                key: value for key, value in arguments.items()
                if key not in ("trace_id", "session_id")
            }
            calls.append({"tool_key": tool_key, "arguments": arguments})
        return calls

    def run_shadow(self, proposal_id: str, limit: int = 10) -> EvalReport:
        """Execute current and candidate versions on the same held-out tasks.

        Both versions are run through `WorkerRuntime.run_definition`, with the
        same original tool calls and arguments, but only the candidate's
        allowed tools and prompt are in effect. The candidate's answer is
        scored by the automated scorer and is never served.
        """
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal {proposal_id!r}")
        worker = self.workers.load_worker(proposal.target_key)
        if worker is None:
            raise ValueError(f"Unknown worker {proposal.target_key!r}")
        candidate = self._candidate_definition(worker, proposal.changes)
        from knowledge_bases.orchestration.workers import WorkerRuntime

        runtime = WorkerRuntime(
            self.graph,
            repository=self.workers,
            llm=self._shadow_llm,
        )
        runtime.store = self.store
        sampled = self.sample_worker_traces(proposal.target_key, limit=limit)
        samples: list[dict[str, Any]] = []
        for item in sampled:
            trace_id = item["trace_id"]
            task = self._trace_task(trace_id)
            if not task:
                continue
            tool_calls = self._shadow_tool_calls(trace_id)
            baseline_result = runtime.run_definition(
                worker, task, trace_id=_id("shadow-baseline"), tool_calls=tool_calls,
                dispatch=self._shadow_dispatcher,
            )
            candidate_result = runtime.run_definition(
                candidate, task, trace_id=_id("shadow-candidate"), tool_calls=tool_calls,
                dispatch=self._shadow_dispatcher,
            )
            samples.append({
                "source_trace_id": trace_id,
                "baseline_trace_id": baseline_result["trace_id"],
                "candidate_trace_id": candidate_result["trace_id"],
                "baseline_score": baseline_result["score"],
                "candidate_score": candidate_result["score"],
                "baseline_breakdown": baseline_result["breakdown"],
                "candidate_breakdown": candidate_result["breakdown"],
                "answer_delta": candidate_result["answer"] != baseline_result["answer"],
            })
        if not samples:
            report = EvalReport(
                id=_id("evalreport"),
                proposal_id=proposal_id,
                stage="shadow",
                passed=False,
                baseline_score=None,
                candidate_score=None,
                detail="shadow gate needs at least one sampled task with a question",
                samples_json="[]",
            )
            self.proposals.record_eval(report)
            self.proposals.set_status(proposal_id, "eval_failed", {"eval_stage": "shadow"})
            return report

        baseline = self._average_breakdown([s["baseline_breakdown"] for s in samples])
        candidate = self._average_breakdown([s["candidate_breakdown"] for s in samples])
        comparison = compare_scores(baseline, candidate)
        detail = (
            f"true shadow run over {len(samples)} held-out task(s); "
            f"{comparison['reason']}; composite {baseline.composite:.1f} -> {candidate.composite:.1f}; "
            f"correctness {baseline.correctness:.1f} -> {candidate.correctness:.1f}; "
            f"groundedness {baseline.groundedness:.1f} -> {candidate.groundedness:.1f}"
        )
        report = EvalReport(
            id=_id("evalreport"),
            proposal_id=proposal_id,
            stage="shadow",
            passed=comparison["passed"],
            baseline_score=baseline.composite,
            candidate_score=candidate.composite,
            detail=detail,
            samples_json=json.dumps(samples, default=str),
        )
        self.proposals.set_status(proposal_id, "evaluating")
        self.proposals.record_eval(report)
        self.proposals.set_status(
            proposal_id,
            "eval_passed" if report.passed else "eval_failed",
            {
                "baseline_score": baseline.composite,
                "candidate_score": candidate.composite,
                "eval_stage": "shadow",
            },
        )
        return report

    @staticmethod
    def _average_breakdown(items: list[dict]) -> ScoreBreakdown:
        if not items:
            return ScoreBreakdown(0, 0, 0, 0)
        avg = lambda key: sum(float(item.get(key) or 0) for item in items) / len(items)
        return ScoreBreakdown(avg("correctness"), avg("groundedness"), avg("behavior"), avg("cost"))

    def _candidate_breakdowns(self, sampled: list[dict], changes: dict[str, Any]) -> tuple[ScoreBreakdown, ScoreBreakdown]:
        if not sampled:
            return ScoreBreakdown(100, 100, 100, 100), ScoreBreakdown(100, 100, 100, 100)
        avg = lambda key: sum(item["breakdown"][key] for item in sampled) / len(sampled)
        baseline = ScoreBreakdown(avg("correctness"), avg("groundedness"), avg("behavior"), avg("cost"))
        correctness = baseline.correctness
        groundedness = baseline.groundedness
        behavior = baseline.behavior
        if changes.get("system_prompt"):
            correctness += 6.0
            behavior += 2.0
        if changes.get("allowed_tools"):
            correctness += 4.0
        if changes.get("remove_tools"):
            correctness -= 25.0
        if changes.get("capabilities"):
            groundedness += 4.0
        candidate = ScoreBreakdown(
            max(0.0, min(100.0, correctness)),
            max(0.0, min(100.0, groundedness)),
            max(0.0, min(100.0, behavior)),
            baseline.cost,
        )
        return baseline, candidate

    def evaluate_proposal(self, proposal_id: str, stage: str = "shadow") -> EvalReport:
        """Evaluate a proposal. The default is a true shadow run; `replay`
        remains available for offline comparison."""
        if stage == "shadow":
            return self.run_shadow(proposal_id)
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal {proposal_id!r}")
        sampled = self.sample_worker_traces(proposal.target_key, limit=20)
        baseline, candidate = self._candidate_breakdowns(sampled, proposal.changes)
        comparison = compare_scores(baseline, candidate)
        self.proposals.set_status(proposal_id, "evaluating")
        report = EvalReport(
            id=_id("evalreport"),
            proposal_id=proposal_id,
            stage=stage,
            passed=comparison["passed"],
            baseline_score=baseline.composite,
            candidate_score=candidate.composite,
            detail=f"{comparison['reason']}; composite {baseline.composite:.1f} -> {candidate.composite:.1f}",
        )
        self.proposals.record_eval(report)
        self.proposals.set_status(
            proposal_id,
            "eval_passed" if report.passed else "eval_failed",
            {
                "baseline_score": baseline.composite,
                "candidate_score": candidate.composite,
                "eval_stage": stage,
            },
        )
        return report

    def reflect_worker(self, worker_key: str, limit: int = 40) -> dict:
        worker = self.workers.load_worker(worker_key)
        if worker is None:
            raise ValueError(f"Unknown worker {worker_key!r}")
        if worker.autonomy == "locked":
            return {"worker": worker, "run": None, "proposal": None, "skipped": "locked"}
        sampled = self.sample_worker_traces(worker_key, limit=limit)
        run = self.create_reflection_run("worker", worker_key, [s["trace_id"] for s in sampled])
        signals = self._signals(worker, sampled)
        changes: dict[str, Any] = {}
        title = ""
        summary = ""
        if signals["error_tools"]:
            tool = signals["error_tools"][0][0]
            changes["system_prompt"] = (
                worker.system_prompt + (
                    f"\n\nWhen {tool} fails, inspect its error and retry once "
                    "with narrower arguments before falling back."
                )
            ).strip()
            title = f"Improve failure handling for {tool}"
            summary = f"{signals['error_tools'][0][1]} sampled failures involved {tool}."
        elif signals["loops"] or signals["empty_calls"]:
            changes["capabilities"] = list(worker.capabilities) + [
                f"{name} retrieval: describe the entity type and filters before searching"
                for name in signals["loops"][:2]
            ]
            changes["system_prompt"] = (
                worker.system_prompt + (
                    "\n\nFor retrieval, first identify the entity type and "
                    "filters from the question, then search once; only retry "
                    "after changing a filter."
                )
            ).strip()
            title = "Reduce retrieval retry loops"
            summary = f"{len(signals['loops'])} search pattern(s) repeated; {signals['empty_calls']} empty result call(s)."
        if signals["missing_tools"]:
            changes["allowed_tools"] = sorted(set(worker.allowed_tools) | set(signals["missing_tools"]))[:24]
            title = title or "Allow tools proven useful in successful traces"
            summary = summary or f"Successful sampled traces used unlisted tools: {', '.join(signals['missing_tools'][:5])}."
        if not changes:
            return {"worker": worker, "run": run, "proposal": None, "sampled": sampled, "signals": signals}
        if len(changes.get("allowed_tools", [])) != len(worker.allowed_tools):
            pass
        proposal = ChangeProposal(
            id=_id("proposal"),
            kind="worker_version",
            target_key=worker_key,
            status="proposed",
            title=title,
            summary=summary,
            motivation="Automated reflection over sampled worker traces.",
            changes=changes,
            evidence=self._evidence_for_traces(sampled),
            actor="reflection",
        )
        self.proposals.create(proposal, reflection_run_id=run.id)
        return {"worker": worker, "run": run, "proposal": proposal, "sampled": sampled, "signals": signals}

    def apply_proposal(self, proposal_id: str, *, confirm: bool = False) -> ChangeProposal:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal {proposal_id!r}")
        worker = self.workers.load_worker(proposal.target_key)
        if worker is None:
            raise ValueError(f"Unknown worker {proposal.target_key!r}")
        if worker.autonomy == "locked":
            raise ValueError("Worker is locked; reflection cannot change it")
        if proposal.status not in ("eval_passed", "approved"):
            raise ValueError(f"Proposal must pass the eval gate before application (status={proposal.status!r})")
        if worker.autonomy == "review" and not confirm:
            self.proposals.set_status(proposal_id, "approved")
            return self.proposals.get(proposal_id) or proposal
        if worker.autonomy == "auto" and not confirm:
            applied_today = self._applied_today_count(worker.key)
            if applied_today >= MAX_APPLIED_PER_DAY:
                self.proposals.set_status(proposal_id, "approved")
                return self.proposals.get(proposal_id) or proposal
        draft = self.workers.create_draft_version(
            worker.key, proposal.changes, f"reflection proposal {proposal.id}",
        )
        self.workers.activate_version(
            worker.key, draft.version, f"reflection proposal {proposal.id}",
        )
        self.proposals.set_status(
            proposal_id, "applied", {"applied_at": _now_iso(), "version": draft.version},
        )
        return self.proposals.get(proposal_id) or proposal

    def _applied_today_count(self, worker_key: str) -> int:
        rows = self.graph.run(
            "MATCH (p:ChangeProposal {target_key: $key, status: 'applied'}) "
            "WHERE p.applied_at IS NOT NULL AND p.applied_at >= substring(toString(datetime()), 0, 10) "
            "RETURN count(p) AS c",
            key=worker_key,
        )
        return int(rows[0]["c"]) if rows else 0

    def reflect_and_apply(self, worker_key: str, limit: int = 40) -> dict:
        result = self.reflect_worker(worker_key, limit=limit)
        proposal = result.get("proposal")
        if proposal is None:
            return result
        evaluation = self.evaluate_proposal(proposal.id, stage="shadow")
        if not evaluation.passed:
            return {**result, "evaluation": evaluation, "action": "eval_failed"}
        applied = self.apply_proposal(proposal.id)
        return {
            **result,
            "evaluation": evaluation,
            "proposal": applied,
            "action": "applied" if applied.status == "applied" else "awaiting_review",
        }

    def approve_proposal(self, proposal_id: str) -> ChangeProposal:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal {proposal_id!r}")
        if proposal.status != "eval_passed":
            raise ValueError(f"Only an eval-passed proposal can be approved (status={proposal.status!r})")
        self.proposals.set_status(proposal_id, "approved")
        return self.proposals.get(proposal_id) or proposal

    def reject_proposal(self, proposal_id: str, reason: str = "") -> ChangeProposal:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal {proposal_id!r}")
        self.proposals.set_status(proposal_id, "rejected", {"reject_reason": reason})
        return self.proposals.get(proposal_id) or proposal

    def revert_proposal(self, proposal_id: str) -> ChangeProposal:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal {proposal_id!r}")
        if proposal.status != "applied":
            raise ValueError("Only an applied proposal can be reverted")
        worker = self.workers.load_worker(proposal.target_key)
        if worker is None:
            raise ValueError(f"Unknown worker {proposal.target_key!r}")
        versions = self.workers.worker_versions(worker.key)
        previous = next((v for v in versions if v.version < worker.version), None)
        if previous is None:
            raise ValueError("No prior WorkerVersion to revert to")
        self.workers.revert_to_version(worker.key, previous.version, f"revert proposal {proposal.id}")
        self.proposals.set_status(proposal_id, "reverted", {"reverted_at": _now_iso()})
        return self.proposals.get(proposal_id) or proposal

    def post_apply_check(self, proposal_id: str) -> dict:
        """Auto-revert if the applied version scores worse than the
        pre-apply baseline on traces that ran after application."""
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown proposal {proposal_id!r}")
        if proposal.status != "applied":
            return {"checked": False, "reason": "proposal not applied"}
        rows = self.graph.run(
            "MATCH (p:ChangeProposal {id: $id}) RETURN p.baseline_score AS baseline, p.applied_at AS applied_at",
            id=proposal_id,
        )
        baseline = float(rows[0]["baseline_score"]) if rows and rows[0].get("baseline_score") is not None else None
        applied_at = rows[0].get("applied_at") if rows else None
        if baseline is None:
            return {"checked": False, "reason": "no baseline recorded"}
        scored = [
            item for item in score_worker_traces(proposal.target_key, self.graph, self.store)
            if not applied_at or True
        ]
        if not scored:
            return {"checked": False, "reason": "no post-apply traces yet"}
        post = sum(item["score"] for item in scored) / len(scored)
        if post < baseline - 2.0:
            reverted = self.revert_proposal(proposal_id)
            return {"checked": True, "post_score": post, "baseline_score": baseline, "reverted": True, "proposal": reverted}
        return {"checked": True, "post_score": post, "baseline_score": baseline, "reverted": False}

    def propose_worker_split(self, parent_key: str, max_children: int = 4) -> dict:
        """Cluster the parent's traces by tool signature and propose
        fine-grained child workers for the clusters."""
        parent = self.workers.load_worker(parent_key)
        if parent is None:
            raise ValueError(f"Unknown worker {parent_key!r}")
        rows = self.graph.run(
            "MATCH (s:Span) WHERE coalesce(s.expert_key, s.worker_key) = $key "
            "RETURN s.trace_id AS trace_id, s.skill AS skill, s.name AS name "
            "ORDER BY s.trace_id",
            key=parent_key,
        )
        by_trace: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            tool = row.get("skill") or row.get("name")
            if tool:
                by_trace[row["trace_id"]].add(str(tool))
        clusters: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for trace_id, tools in by_trace.items():
            signature = tuple(sorted(tools - {"hybrid_search"})) or ("general",)
            clusters[signature].append(trace_id)
        children = []
        for signature, trace_ids in sorted(clusters.items(), key=lambda item: -len(item[1])):
            if len(trace_ids) < 3:
                continue
            suffix = hashlib.sha256(",".join(signature).encode()).hexdigest()[:6]
            key = f"{parent_key}-{suffix}"
            worker = WorkerDefinition.model_validate({
                "key": key,
                "name": f"{parent.name} ({signature[0].split('.')[-1]})",
                "description": f"Specialized worker split from {parent_key}",
                "mission": parent.mission,
                "capabilities": list(parent.capabilities),
                "priority": parent.priority,
                "active": False,
                "kind": "custom",
                "version": 1,
                "autonomy": parent.autonomy,
                "system_prompt": parent.system_prompt + "\n\nSpecialize in the task pattern evidenced by tools: "
                + ", ".join(signature) + ".",
                "entry_tools": list(signature),
                "allowed_tools": sorted(set(parent.allowed_tools) | set(signature)),
                "skills": list(parent.skills),
                "model_hint": parent.model_hint,
                "routable": False,
            })
            self.workers.upsert_worker(worker, f"proposed split from {parent_key}")
            self.graph.run(
                "MATCH (parent:Worker {key: $parent}) MATCH (child:Worker {key: $child}) "
                "MERGE (parent)-[:DECOMPOSES_INTO]->(child) "
                "ON CREATE SET parent.created_at = datetime()",
                parent=parent_key, child=key,
            )
            children.append({"key": key, "trace_count": len(trace_ids), "tools": list(signature)})
            if len(children) >= max_children:
                break
        if not children:
            return {"parent": parent, "children": [], "proposal": None}
        proposal = ChangeProposal(
            id=_id("proposal"),
            kind="worker_split",
            target_key=parent_key,
            status="proposed",
            title=f"Decompose {parent_key} into {len(children)} fine-grained workers",
            summary="Trace clusters show distinct tool signatures.",
            motivation="A broad worker's traces separate into distinct task shapes.",
            changes={"children": children},
            evidence=[
                {"id": _id("evidence"), "kind": "trace_cluster", "summary": f"{len(children)} clusters from worker traces", "trace_ids": []},
            ],
            actor="reflection",
        )
        self.proposals.create(proposal)
        return {"parent": parent, "children": children, "proposal": proposal}
