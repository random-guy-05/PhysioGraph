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

## 3. THE "PERMISSION GATE" PROTOCOL
- **Discovery Phase:** Identify files requiring changes.
- **Confirmation Halt:** List target files and the specific reason for selection.
- **Wait:** You must receive a user `[ACK]` or confirmation before reading the full content of those files or executing writes.

## 4. AGENT ROLE: SISYPHUS (EXECUTION)
- **Constraint:** Max 10 tool calls per turn.
- **State Management:** Maintain a minimalist scratchpad: `[CURRENT_STEP]`, `[REMAINING_STEPS]`, `[PENDING_CHANGES]`.
- **Context Purge:** Do not repeat previous "thoughts." Focus only on the immediate delta and next action.
- **Loop Prevention:** If a tool fails twice with the same error, HALT and report. Do not retry.

## 5. AGENT ROLE: ORACLE (ARCHITECT)
- **Direct Entry:** Start responses with the technical solution. No preambles like "I have analyzed the code."
- **Logic Mapping:** Verify proposed logic against `package.json` or existing config files before outputting.
- **Verification:** Every proposal must include a single-line test command to verify the fix.

## 6. ERROR & FAILURE SYNTAX
- **Constraint Violation:** If a task is impossible under these token-saving rules, output: `[ERROR: CONSTRAINT_VIOLATION] - <reason>`.
- **Ambiguity:** If the user request is vague, ask exactly ONE clarifying question.
