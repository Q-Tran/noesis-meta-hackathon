// Headless smoke test for web/index.html — no browser, no jsdom.
//
// `node --check` catches syntax errors only. This harness stubs just enough
// DOM for the SPA to boot, then exercises its views and asserts the graph
// explorer constructs and merges data — the regression guard for boot-time
// runtime errors (two of which have shipped). Run: node scripts/spa_smoke.mjs

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { runInThisContext } from "node:vm";
import assert from "node:assert/strict";

const repo = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(repo, "web", "index.html"), "utf8");
const script = html.match(/<script>\n([\s\S]*)\n<\/script>/)[1];

/* ---------- DOM stubs ---------- */

class ClassList {
  constructor() { this.set = new Set(); }
  add(...cls) { cls.forEach(c => this.set.add(c)); }
  remove(...cls) { cls.forEach(c => this.set.delete(c)); }
  toggle(cls, force) {
    const want = force === undefined ? !this.set.has(cls) : !!force;
    if (want) this.set.add(cls); else this.set.delete(cls);
    return want;
  }
  contains(cls) { return this.set.has(cls); }
}

const ctxStub = new Proxy({}, { get: () => () => undefined, set: () => true });

class FakeElement {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.classList = new ClassList();
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this._innerHTML = "";
    this._textContent = "";
    this._listeners = {};
    this._id = "";
    this.value = "";
    this.clientWidth = 960;
    this.clientHeight = 600;
  }
  get className() { return [...this.classList.set].join(" "); }
  set className(v) { this.classList.set = new Set(String(v).split(" ").filter(Boolean)); }
  get id() { return this._id; }
  set id(v) { this._id = v; }
  set innerHTML(v) {
    this._innerHTML = v;
    this.children = [];
    // Keep querySelector working for the classes the markup declares.
    for (const m of String(v).matchAll(/class="([^"]+)"/g)) {
      for (const cls of m[1].split(" ")) {
        if (cls && !this.children.some(c => c.classList.contains(cls))) {
          const child = new FakeElement("div");
          child.classList.add(cls);
          this.appendChild(child);
        }
      }
    }
  }
  get innerHTML() { return this._innerHTML; }
  set textContent(v) { this._textContent = v; this.children = []; }
  get textContent() { return this._textContent; }
  insertAdjacentHTML(_pos, v) { this._innerHTML += v; }
  setAttribute(k, v) { this.attributes[k] = String(v); }
  getAttribute(k) { return this.attributes[k]; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  append(...cs) { for (const c of cs) this.appendChild(c); }
  insertBefore(c, ref) {
    c.parentNode = this;
    const i = ref ? this.children.indexOf(ref) : -1;
    if (i < 0) this.children.push(c); else this.children.splice(i, 0, c);
    return c;
  }
  remove() {
    if (this.parentNode) {
      const i = this.parentNode.children.indexOf(this);
      if (i >= 0) this.parentNode.children.splice(i, 1);
      this.parentNode = null;
    }
  }
  _find(pred) {
    for (const c of this.children) {
      if (pred(c)) return c;
      const r = c._find(pred);
      if (r) return r;
    }
    return null;
  }
  querySelector(sel) {
    if (sel.startsWith("#")) {
      return this._find(c => c.id === sel.slice(1)) || documentStub.querySelector(sel);
    }
    if (sel.startsWith(".")) return this._find(c => c.classList.contains(sel.slice(1)));
    return this._find(c => c.tagName === sel.toUpperCase());
  }
  querySelectorAll(sel) {
    const out = [];
    const cls = sel.replace(/^\./, "");
    (function walk(el) {
      for (const c of el.children) {
        if (c.classList.contains(cls)) out.push(c);
        walk(c);
      }
    })(this);
    return out;
  }
  addEventListener(type, fn) { (this._listeners[type] ||= []).push(fn); }
  removeEventListener() {}
  focus() {}
  click() { this.dispatch("click"); }
  scrollIntoView() {}
  setPointerCapture() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight }; }
  getContext() { return ctxStub; }
  dispatch(type, event = {}) {
    for (const fn of this._listeners[type] || []) {
      fn({ preventDefault() {}, stopPropagation() {}, target: this, ...event });
    }
  }
}

const tabStubs = ["graph", "dashboard", "ask"].map(view => {
  const el = new FakeElement("button");
  el.dataset.view = view;
  return el;
});

const documentStub = {
  byId: new Map(),
  root: new FakeElement("body"),
  createElement: tag => new FakeElement(tag),
  createElementNS: (_ns, tag) => new FakeElement(tag),
  querySelector(sel) {
    if (sel.startsWith("#")) {
      const id = sel.slice(1);
      if (!this.byId.has(id)) this.byId.set(id, new FakeElement("div"));
      return this.byId.get(id);
    }
    const found = this.root.querySelector(sel);
    if (found) return found;
    // Tag selectors resolve to real markup (e.g. <main>); auto-create on miss.
    if (/^[a-z][a-z0-9]*$/i.test(sel)) {
      const el = new FakeElement(sel);
      this.root.appendChild(el);
      return el;
    }
    return null;
  },
  querySelectorAll(sel) {
    if (sel === ".tab") return tabStubs;
    if (sel.startsWith("#sidebar")) {
      return this.byId.get("sidebar")?.querySelectorAll(sel.split(" ").pop()) || [];
    }
    return this.root.querySelectorAll(sel);
  },
  addEventListener() {},
  body: { contains: () => true },
};

/* ---------- browser stubs ---------- */

globalThis.document = documentStub;
globalThis.window = { addEventListener() {}, removeEventListener() {}, devicePixelRatio: 1 };
globalThis.location = { hash: "" };
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
let rafBudget = 6;
globalThis.requestAnimationFrame = cb => { if (rafBudget-- > 0) cb(); return 0; };

/* ---------- fetch stub ---------- */

// The home map: indexed wire form, like /api/graph/subset returns.
const SUBSET = {
  labels_dict: ["Component", "Interface"],
  rel_types: ["USES", "EXPOSES"],
  ids: ["e1", "e5", "e6"],
  nodes: [[0, "hub", 5], [1, "iface a", 2], [0, "comp b", 2]],
  edges: [[0, 0, 1], [1, 0, 2]],
};
const SEED = {
  nodes: [
    { element_id: "e1", labels: ["Component"], title: "hub", degree: 2, properties: {} },
    { element_id: "e2", labels: ["Tool"], title: "tool a", properties: {} },
    { element_id: "e3", labels: ["Chunk"], title: "chunk a", properties: {} },
  ],
  edges: [
    { element_id: "r1", type: "USES", source: "e1", target: "e2", outgoing: true },
    { element_id: "r2", type: "FOUND_IN", source: "e3", target: "e1", outgoing: false },
  ],
};
const FULL = {
  labels_dict: ["Component", "Interface"],
  rel_types: ["USES", "EXPOSES"],
  ids: ["e1", "e2", "e5", "e6", "e7"],
  nodes: [[0, "hub"], [1, "iface a"], [0, "comp b"], [1, "iface b"], [0, "comp c"]],
  edges: [[0, 0, 1], [1, 2, 3], [1, 4, 3]],
};
const ROUTES = [
  ["/api/graph/subset", SUBSET],
  ["/api/graph/full", FULL],
  ["/api/overview", { total_nodes: 3, total_relationships: 2, labels: [{ label: "Component", count: 2 }, { label: "Tool", count: 1 }], relationships: [{ type: "USES", count: 2 }] }],
  ["/api/growth", [{ day: "2026-11-02", count: 1 }, { day: "2026-11-03", count: 2 }]],
  ["/api/activity", []],
  ["/api/search", []],
  ["/api/ask", {
    question: "hi",
    answer: "Sure — **bold** answer.",
    tools: [{ name: "search_corpus" }, { name: "get_entity" }],
    writes: [{ name: "add_claim", claim_id: "chatclaim-x1" }],
    trace_id: "chat-smoke",
    session_id: "web-smoke",
    model: "muse-wonder-1.0",
    entities: [{ id: "extent-ts", title: "Tool search", label: "Capability" }],
    tool_search: [{ searched: { query: "hi" }, loaded: ["search_corpus", "get_entity"] }],
    delta: {
      nodes: [
        { element_id: "dn1", id: "chatclaim-x1", label: "Claim", title: "smoke claim", status: "active", mark: "+" },
        { element_id: "dn2", id: "extarg-old", label: "Argument", title: "old argument", status: "out", mark: "⚔" },
      ],
      edges: [
        { element_id: "dr1", type: "ATTACKS", source: "chatclaim-x1", source_label: "Claim", target: "extarg-old", target_label: "Argument", mark: "⚔" },
      ],
      explanation: [
        { mark: "+", text: "Claim ‘smoke’", why: "you asked for it" },
      ],
    },
  }],
  ["/api/graph/related", {
    centers: ["e-ts"],
    nodes: [
      { element_id: "e-ts", labels: ["Capability"], title: "Tool search", properties: {} },
      { element_id: "e-a1", labels: ["Argument"], title: "an argument", properties: {} },
    ],
    edges: [{ element_id: "rr-ts", type: "CONCLUDES", source: "e-a1", target: "e-ts", outgoing: false }],
  }],
  ["/api/trace/chat-smoke", { trace_id: "chat-smoke", spans: [
    { span_id: "s-1", name: "search_corpus", kind: "tool", status: "ok", duration_ms: 90 },
  ] }],
  ["/api/graph/layer_neighbors", { nodes: [{ element_id: "c1", labels: ["Claim"], title: "claim a", properties: {} }], edges: [{ element_id: "rr1", type: "USES_PREMISE", source: "e1", target: "c1", outgoing: true }], layer: "epistemology" }],
  ["/api/node/e1/neighbors", { nodes: [...SEED.nodes, { element_id: "e4", labels: ["Interface"], title: "iface new", properties: {} }], edges: [...SEED.edges, { element_id: "r3", type: "EXPOSES", source: "e1", target: "e4", outgoing: true }] }],
  ["/api/node/e1", { element_id: "e1", labels: ["Component"], title: "hub", properties: {} }],
];

globalThis.fetch = async (url) => {
  if (String(url).startsWith("/api/ask/stream")) {
    // A real SSE body: progress events first (snapshot the progress rows the
    // UI renders mid-stream), then the final event with the /api/ask payload.
    const askPayload = ROUTES.find(([p]) => p === "/api/ask")[1];
    const enc = new TextEncoder();
    const chunks = [
      enc.encode('data: {"type":"tool_start","name":"search_corpus"}\n\n' +
                 'data: {"type":"tool_done","name":"search_corpus","duration_ms":90,"summary":{"name":"search_corpus","hits":1},"write":false}\n\n'),
      enc.encode(`data: ${JSON.stringify({ type: "final", result: askPayload })}\n\n`),
    ];
    let i = 0;
    return {
      ok: true, status: 200,
      body: {
        getReader: () => ({
          read: async () => {
            if (i === 1) {
              globalThis.__streamSnapshot =
                documentStub.querySelector("#ask-thread").querySelectorAll(".prog-row").length;
            }
            if (i < chunks.length) return { done: false, value: chunks[i++] };
            return { done: true };
          },
        }),
      },
    };
  }
  const hit = ROUTES.find(([p]) => String(url).startsWith(p));
  if (!hit) return { ok: false, status: 404, json: async () => ({ detail: "not found" }) };
  return { ok: true, status: 200, json: async () => hit[1] };
};

/* ---------- run the SPA and exercise it ---------- */

// vm top-level `let` bindings (explorerInstance, setView) live in the vm
// context's global lexical scope, not on globalThis — read them back out
// with runInThisContext.
runInThisContext(script, { filename: "web/index.html<script>" });
const flush = () => new Promise(r => setTimeout(r, 30));
await flush();

const setView = runInThisContext("setView");
const ex = runInThisContext("explorerInstance");
assert.ok(ex, "graph explorer constructed at boot");

// Boot merged the home subset (all canonical, all interrelated).
assert.equal(ex.nodeById.size, 3, `home subset merged (got ${ex.nodeById.size})`);
assert.equal(ex.edges.length, 2, "subset edges among the chosen nodes");
assert.ok(ex.nodeById.get("e1").r > 7, "the hub reads as a hub (degree-sized)");

// The graph view attributes the corpus to its ingestion model.
assert.ok(documentStub.byId.get("sidebar")?.querySelectorAll(".map-model").length >= 1,
  "sidebar carries the Muse Spark ingestion chip");

// Expansion merges the new canonical neighbors; provenance still filters out.
await ex.expand(ex.nodeById.get("e1"));
assert.ok(ex.nodeById.has("e4"), "expand merged the new neighbor");
assert.ok(ex.nodeById.has("e2"), "the Tool neighbor is canonical now (product capability)");
assert.ok(!ex.nodeById.has("e3"), "the Chunk (provenance) stays out");
assert.equal(ex.nodeById.size, 5, "5 nodes after expand");

// Toggling epistemology ON populates the map with the justification ring;
// toggling it back OFF removes that layer again.
await ex.toggleLayer("epistemology");
assert.ok(ex.state.layers.epistemology, "epistemology pill is on");
assert.ok(ex.nodeById.has("c1"), "toggle-on populated the claim");
await ex.toggleLayer("epistemology");
assert.ok(!ex.nodeById.has("c1"), "toggle-off removed the claim");

// Load all replaces the graph and switches to the community layout.
await ex.loadFull();
assert.equal(ex.state.mode, "static", "loadFull switched to static layout");
assert.equal(ex.nodeById.size, 5, "loadFull rehydrated the indexed payload");
assert.equal(ex.groupLabels.length, 2, "community layout grouped by relationship");

// The graph sidebar speaks about the map: an honest "x of y" subset count
// and the kinds on the map, and its clicks filter the map in place.
const sidebarEl = documentStub.byId.get("sidebar");
const sidebarText = sidebarEl.children[0]._innerHTML.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
assert.ok(sidebarEl.children[0]._innerHTML.includes("This map"),
  "graph sidebar is scoped to the map, not the corpus");
assert.ok(/\d+ of \d+ nodes in the corpus/.test(sidebarText),
  "graph sidebar shows the subset count against the corpus total");
assert.ok(/\d+ of \d+ relationships/.test(sidebarText),
  "relationships count shows the corpus total too");
ex.toggleGraphFilter("Component");
assert.ok(ex.filterLabels.has("Component"), "filter applied");
assert.deepEqual(ex.state.filterLabels, ["Component"], "filter persisted across visits");
ex.toggleGraphFilter("Interface");
assert.deepEqual([...ex.filterLabels].sort(), ["Component", "Interface"],
  "a second click accumulates (union, multi-select)");
assert.equal(runInThisContext("state.view"), "graph",
  "clicking entity kinds keeps you in graph view (no list view)");
ex.toggleGraphFilter("Component");
assert.deepEqual([...ex.filterLabels], ["Interface"],
  "clicking a selected kind drops it from the union");
ex.clearGraphFilters();
assert.equal(ex.filterLabels.size, 0, "clear restores everything");
assert.deepEqual(ex.state.filterLabels, [], "clear persists too");

// Views switch without throwing; explorer state persists across visits.
setView("dashboard");
await flush();
setView("ask");
assert.ok(documentStub.querySelector("main").innerHTML.includes("Muse Wonder"),
  "ask view carries the Muse Wonder chat-model chip");
documentStub.querySelector("#ask-input").value = "hi";
documentStub.querySelector("#ask-form").dispatch("submit");
await flush();
assert.ok(globalThis.__streamSnapshot >= 1, "progress rows rendered while the turn streamed");
// The answer renders chat-botty (user bubble + avatar) with the provenance
// chrome; tool search lives inside the trace toggle, emphasized.
// (The DOM stub parses no innerHTML, so query through the appended elements.)
const thread = documentStub.querySelector("#ask-thread");
assert.ok(thread.querySelectorAll(".q").length >= 1, "user bubble rendered");
assert.ok(thread.querySelectorAll(".avatar").length >= 1, "assistant avatar rendered");
const answers = thread.querySelectorAll(".a");
const answerHtml = answers[answers.length - 1].innerHTML;
assert.ok(!answerHtml.includes("Meta tool search"), "tool search no longer banners the answer");
assert.ok(answerHtml.includes("search_corpus"), "tool chips rendered");
assert.ok(answerHtml.includes("chatclaim-x1"), "write receipt rendered");
assert.ok(answerHtml.includes("graph changes"), "delta card rendered");
const traceBtn = thread.querySelectorAll(".msg-tracebtn").pop();
assert.ok(traceBtn, "trace button rendered");
assert.ok(traceBtn.textContent.includes("tool search"), "trace button names tool search");
assert.ok(traceBtn.textContent.includes("muse-wonder-1.0"), "trace button names the chat model");
traceBtn.dispatch("click");
await flush();
const tracePanel = thread.querySelectorAll(".msg-trace").pop();
assert.ok(tracePanel.innerHTML.includes("Meta tool search"), "tool search emphasized inside the trace");
assert.ok(tracePanel.innerHTML.includes("search_corpus"), "trace toggle loaded the spans");
// Both mini maps mount a canvas: the delta visual and the related entities.
const dv = thread.querySelectorAll(".delta-visual").pop();
assert.ok(dv && dv.children.some(c => c.tagName === "CANVAS"), "delta visual canvas mounted");
await flush();
const relatedCard = thread.querySelectorAll(".mini-graph").pop();
assert.ok(relatedCard && relatedCard.children.some(c => c.tagName === "CANVAS"),
  "related-entities mini map canvas mounted");
setView("graph");
await flush();
assert.equal(runInThisContext("explorerInstance").nodeById.size, 5,
  "explorer state persisted across tab visits");

// Layer toggle removes the disabled layer's nodes (the all-canonical map
// empties); Reset returns to the home subset.
const ex2 = runInThisContext("explorerInstance");
ex2.toggleLayer("canonical");
assert.equal(ex2.nodeById.size, 0, "canonical toggle-off emptied the all-canonical map");
ex2.toggleLayer("canonical"); // back on; the reset load respects it again
await ex2.resetHome();
assert.equal(ex2.nodeById.size, 3, "reset re-loaded the home subset");
assert.equal(ex2.state.mode, "force", "reset returned to the force layout");

// Deep links: #n=<eid> opens that node; #view=<tab> opens that tab. The
// second boot is wrapped in an IIFE that returns the resulting state.view, so
// the script's top-level lets re-declare cleanly each run.
globalThis.location.hash = "#n=e1";
let resultView = runInThisContext(`(() => { ${script}; return state.view; })()`, { filename: "web/index.html<script>#2" });
await flush();
assert.equal(resultView, "node", "hash #n= opened the node detail");
globalThis.location.hash = "#view=ask";
resultView = runInThisContext(`(() => { ${script}; return state.view; })()`, { filename: "web/index.html<script>#3" });
await flush();
assert.equal(resultView, "ask", "hash #view= opened the ask tab");
globalThis.location.hash = "#view=dashboard";
resultView = runInThisContext(`(() => { ${script}; return state.view; })()`, { filename: "web/index.html<script>#4" });
await flush();
assert.equal(resultView, "dashboard", "hash #view= opened the dashboard tab");
globalThis.location.hash = "";

console.log("SPA smoke test passed: boot, subset, expand, load-all, views, ask, layers, reset, deep-links");
