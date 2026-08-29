## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

# PhysioGraph Project Instructions

- The user's primary runnable deliverable is ONE clean Google Colab notebook that runs start-to-finish.
- Package modules may exist as implementation support, but the runnable workflow must be reflected in an actual `.ipynb` notebook.
- Do not claim code was run in Google Colab unless it actually ran in Google Colab.
- Metrics derived from local or precomputed outputs must be labeled as local/precomputed, not as fresh Colab execution.
- Avoid producing scattered `.py` files as the only usable output; notebook usability is the acceptance target.

# CORE EXECUTION PROTOCOL: AGENTS.MD

## 1. TOKEN PRESERVATION & MECHANICAL CONSTRAINTS
- **Snippets Only:** NEVER rewrite entire files. Use `diff` format or line-marked snippets. Full-file output is a violation.
- **Silent Mode:** Omit all conversational fillers, apologies, and step-by-step pre-planning. Move immediately to tool calls or code.
- **Context Minimalism:** Read only relevant methods or line ranges. Do not perform global searches (`grep -r`, `ls -R`) unless the request is fundamentally ambiguous.
- **Command Bundling:** Chain terminal commands (e.g., `cd path && npm run build`) to minimize turn-over overhead.

## 2. FILESYSTEM & GIT ISOLATION (STRICT)
- **Gitignore Firewall:** Strictly respect `.gitignore`. Never `read`, `search`, or `grep` ignored paths.
- **Validation:** If a path's status is unclear, run `git check-ignore <path>` before interacting.
- **Discovery:** Use `git ls-files` for project mapping. Exclude all ignored paths from logic reasoning.
- **Boundary Warning:** If an error originates in an ignored file (e.g., `node_modules`), report the path to the user and HALT. Do not investigate.

## 3. ERROR & FAILURE SYNTAX
- **Constraint Violation:** If a task is impossible under these token-saving rules, output: `[ERROR: CONSTRAINT_VIOLATION] - <reason>`.
- **Ambiguity:** If the user request is vague, ask exactly ONE clarifying question.
