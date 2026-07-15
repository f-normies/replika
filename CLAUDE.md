# CLAUDE.md

Guide for AI who work this repo.

## MANDATORY: use superpowers every session

This repo run **superpowers** workflow. Session start, and before every task, check for matching skill and invoke it **before acting** — even before clarify question or explore code. Process skill set approach; implementation skill do it. Unsure if skill apply? Assume yes, invoke.

- **Build / add / change behavior** → `superpowers:brainstorming` first (design + your approval), then `superpowers:writing-plans`, then build.
- **Any bug / test failure / unexpected behavior** → `superpowers:systematic-debugging` before propose fix.
- **Implement feature or bugfix** → `superpowers:test-driven-development` (write test first).
- **Before claim done / commit / open PR** → `superpowers:verification-before-completion` (run it, show proof).
- **Finish work / before merge** → `superpowers:requesting-code-review`.
- Isolated work → `superpowers:using-git-worktrees`; 2+ independent task → `superpowers:dispatching-parallel-agents`.

This override ad-hoc action. Combine with context7 rule below (know third-party context before design or code).

## MANDATORY: verify third-party APIs with context7

Before touch **any** library, framework, SDK, CLI, or tool you not write in this repo, you MUST query **context7 MCP** to confirm current API, version, setup. No exception.

- Apply to **all** subsystem.
- Apply to **every** work: new feature, bug fix, refactor, dependency bump, and **before** brainstorm or write implementation plan.
- Do **not** trust training memory for third-party API/version — it drift. Confirm context7 first, then act. Web search only to supplement.

Own code in this repo no need context7 — read it direct.

## MANDATORY: use swift-lsp for Swift file

This repo is native SwiftUI app — Swift is main language. **swift-lsp** plugin (SourceKit-LSP) enable in `.claude/settings.json`. Touch **any** `.swift` file → use **`LSP` tool** for symbol work. Grep is fallback, not first move.

- **Find symbol** → `workspaceSymbol` (always pass `query`). **Where define** → `goToDefinition`. **Who call it** → `findReferences`. **Type / doc** → `hover`. **File outline** → `documentSymbol`.
- **Who conform to protocol** (`TranscriptionProvider`, `TextProvider`) → `goToImplementation`. Grep miss conformance in extension — LSP not.
- **Trace call path** → `prepareCallHierarchy`, then `incomingCalls` / `outgoingCalls`.
- Before rename or change signature of Swift symbol → run `findReferences` first, know every call site.
- Grep still fine for non-symbol text: string literal, comment, config, `.md`.

Gotcha:
- `LSP` tool may be deferred — load with `ToolSearch` query `select:LSP` before first call.
- `line` / `character` are **1-based** (like editor show), not 0-based like raw LSP protocol.
- SourceKit-LSP need index. Fresh clone or brand-new file → build first (`swift build` / Xcode), else cross-file lookup come back empty.

## Keep this file current (run claude-md-improver)

Invoke **`claude-md-improver`** skill to audit + update this file when trigger below fire. Then review its report, apply approved targeted edit, and commit as `docs:`. (These event trigger checked during work — no cron; honor them as they occur.)

Trigger — run skill after any of such examples:
- **Dependency / SDK / tooling change**: version bump, add/remove library, package-manager / linter / formatter / type-checker / CI change.
- **Structure change**: new/removed subsystem, module, or directory convention; changed layer/boundary or config-driven pattern.
- **Workflow change**: `Makefile` target, GitHub Actions, git/branch/commit convention, or pre-commit hook.
- **New non-obvious gotcha**: any footgun that cost real debug time.
- **Milestone boundary**: before merge significant feature branch or close work item.
- **Staleness check**: at session start, if none above fired but ≥10 code commit landed since this file last touched (`git log --oneline -1 -- CLAUDE.md` vs `git log --oneline`), do quick pass.

Keep addition minimal and project-specific (follow skill own rule).