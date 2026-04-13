# Claude Code Rules - Main Repository

## Mission

Upgrade the manuscript in `paper/` toward publication quality by following
`docs/revision_suggestions.tex`, or by running an independent venue-driven
review reset when explicitly requested by the user, modifying the manuscript,
adding experiments when needed, generating figures or tables, integrating them
into the paper, and updating `docs/progress.md`.

This repository may also be used as a Claude rule template that is copied into
another repository. In the template repo itself, canonical destination-repo
files such as `docs/revision_suggestions.tex`, `docs/progress.md`,
`paper/main.tex`, `paper/appendix.tex`, and `paper/references.bib` may be
absent. Treat those paths as the expected layout of the destination repo rather
than as a template error.

The repository-wide process may use multiple agents in parallel when helpful,
and it should keep iterating automatically across revision cycles until the
current `docs/revision_suggestions.tex` has no remaining actionable items.

The agent should work autonomously and iteratively until all actionable
revision items are either:
1. fully completed, or
2. explicitly marked as blocked or partially completed with reasons and next steps.

---

## Repository Layout

- Shared agent entry file: `USAGE.md`
- Target venue configuration: `USAGE.md`
- Revision guidance: `docs/revision_suggestions.tex`
- Progress tracking: `docs/progress.md`

- Main manuscript: `paper/main.tex`
- Appendix: `paper/appendix.tex`
- Bibliography: `paper/references.bib`
- Figures: `paper/figs/`
- Build script: `paper/build.bat`
- Build outputs: `paper/build/`

- Experiment and implementation code: `src/`

---

## Language Rules

### English-only locations
The following must be written in English only:
- `docs/revision_suggestions.tex`
- all files under `src/`
- all code
- all comments
- all docstrings
- all logs
- all config descriptions
- all file names created under `src/`
- all generated experiment summaries saved under `src/`
- all manuscript content under `paper/`
- all captions, table notes, titles, and scientific analysis
- all BibTeX entries in `paper/references.bib`

### Chinese-language exceptions
`docs/progress.md` may contain Chinese.
`USAGE.md` may contain Chinese.

No Chinese is allowed anywhere else unless the user explicitly requests it.

---

## Reader-first Manuscript Boundary

The manuscript must be understandable to reviewers and readers without any
knowledge of the local repository layout or implementation file organization.

Do not mention in `paper/`:
- local code paths
- directory names such as `src/experiments/`, `src/figures/`, or similar repo-internal folders
- script names such as `src/run_all.py`
- concrete code filenames used only for local implementation
- instructions about where files are stored in the repository

When experiments or methods are described in the manuscript, rewrite them in
reader-facing scientific prose. Describe the method, motivation, setup,
metrics, evidence, and conclusions, not the local file layout used to produce
them.

---

## Venue Target

The authoritative target venue should be read from the line
`目标会议或期刊：...` in `USAGE.md`, under the `## 目标会议和期刊` section.

If that line is filled in:
- use it as the primary venue for review standards, scope checks, formatting expectations, and submission-readiness judgments
- align manuscript revisions and experiment expectations to that venue
- verify current official venue requirements when they may be time-sensitive

If that line is blank or unavailable, infer the strongest reasonable top-tier
venue standard from the manuscript and repository context.

---

## Global Workflow

1. Read `docs/revision_suggestions.tex`, unless the user explicitly triggers the
   independent review reset command described below.
2. Audit the current manuscript from source and rendered artifacts:
   - `paper/main.tex`
   - `paper/appendix.tex`
   - `paper/main.pdf` when available
   - `paper/appendix.pdf` when available
   - `paper/references.bib`
3. Build an actionable revision plan mapped to concrete files and sections.
4. Use one agent or multiple agents with clear ownership, depending on the workload.
5. Revise the paper section by section.
6. If a revision item requires experiments:
   - implement or extend code in `src/`
   - run experiments
   - save final paper-ready figures to `paper/figs/`
   - insert figures or tables into the manuscript
   - add corresponding result description and analysis in the paper
   - ensure the manuscript describes scientific setup and findings without exposing local code paths, folder names, or filenames
7. Re-check `docs/revision_suggestions.tex` after each substantial batch of edits.
8. Continue iterating until all actionable items are resolved or explicitly documented as blocked.
9. If only non-actionable or genuinely blocked items remain after best effort, start a new independent review cycle:
   - ignore prior review history and prior completion judgments
   - review the current paper directly from `.tex` and `.pdf` content
   - assess it against the target venue from `USAGE.md` when available, otherwise use the strongest reasonable top-tier venue standard
   - overwrite `docs/revision_suggestions.tex` completely with the new English-only LaTeX review
   - begin a fresh autonomous revision cycle against that rewritten file
10. Update `docs/progress.md` in Chinese to reflect the actual repository state.

---

## Multi-Agent Execution Policy

Multi-agent execution is allowed and encouraged when it materially reduces the
time needed for a revision cycle.

When using multiple agents:
- assign disjoint ownership whenever possible, such as manuscript sections, experiment pipelines, figures, references, or reproducibility checks
- keep the critical path local if the next decision depends on it immediately
- do not let one agent silently overwrite or revert another agent's useful work
- integrate results back into one coherent manuscript state
- re-run the repository-wide checks after parallel work is merged

---

## Automatic Revision Loop

The default behavior is not a single pass.

The repository process must repeatedly:
- read the current `docs/revision_suggestions.tex`
- implement all actionable manuscript, experiment, figure, and citation changes
- re-audit the updated paper
- continue until the current review file has been exhausted as far as the repository can support

If an item remains because of missing data, impossible verification, external resource limits, or a real scientific block, treat it as non-actionable for the current cycle and move to the fresh review protocol below rather than stopping permanently.

---

## Fresh Independent Review Protocol

When the current `docs/revision_suggestions.tex` has no remaining actionable items and only blocked or non-actionable issues remain:

1. Start a new paper review from scratch.
2. Do not rely on previous review notes, prior completion status, or earlier revision judgments.
3. Review the current manuscript directly from:
   - `paper/main.tex`
   - `paper/appendix.tex`
   - `paper/main.pdf` when available
   - `paper/appendix.pdf` when available
4. Evaluate the paper against the target venue declared in `USAGE.md` when available. If no target venue is declared, use the strongest reasonable top-tier venue standard that can be inferred from the manuscript and repository.
5. Rewrite `docs/revision_suggestions.tex` by deleting the old review content and replacing it completely with a new English-only LaTeX review.
6. Make the new review self-contained and independent of earlier review rounds.
7. Start the next autonomous revision cycle using only the newly written `docs/revision_suggestions.tex`.

---

## Independent Review Reset Command

If the user says `重新开始评审并生成评审修改意见`, interpret it as a full independent review reset command.

When this command is triggered:
1. Do not use the current `docs/revision_suggestions.tex` as review input.
2. Read the target venue from the line `目标会议或期刊：...` in `USAGE.md`.
3. Review the manuscript directly from:
   - `paper/main.tex`
   - `paper/appendix.tex`
   - `paper/main.pdf` when available
   - `paper/appendix.pdf` when available
4. Evaluate the paper against the declared target venue. If no target venue is declared, infer the strongest reasonable top-tier venue standard from the repository context.
5. Check contribution quality, scope fit, formatting expectations, evaluation strength, and submission readiness against that venue.
6. Rewrite `docs/revision_suggestions.tex` completely in English-only LaTeX.
7. Ensure the rewritten file contains both review findings and concrete revision actions.
8. After rewriting the file, begin the next autonomous revision cycle against the new review unless the user explicitly asked to stop after the review rewrite.

---

## Autonomy Rule

The agent should act autonomously and should not ask for trivial confirmation.

Only ask the user when a decision materially affects scientific direction and cannot be resolved from the repository.

Examples of valid questions:
- choosing between incompatible experiment designs
- selecting one of two conflicting paper narratives
- missing data or credentials
- an extremely expensive experiment that should not be run without confirmation

When asking, provide:
1. the exact decision point
2. the options
3. the recommended option
4. why it matters

---

## Scientific Quality Target

Target standard: the venue named in `USAGE.md` when available, otherwise the strongest reasonable top-tier journal or conference standard that can be inferred from `docs/revision_suggestions.tex`, the manuscript, and repository context.

The final manuscript should aim for:
- one consistent problem formulation
- coherent notation and method description
- fair and strong experiments
- sufficient baselines and ablations
- privacy-utility evidence where relevant
- reproducibility details
- publication-ready figures and tables
- polished academic English

---

## fal.ai Image Redrawing Rules

When using fal.ai to redraw or regenerate images — especially flowcharts,
framework diagrams, architectural illustrations, and other schematic figures —
the following rules apply:

1. **Model**: Always use the **Nano Banana** model on fal.ai. Do not use other
   fal.ai models for image redrawing tasks.
2. **Preserve all core elements**: The redrawn image must retain every core
   element from the original — all boxes, arrows, labels, layers, modules, data
   flows, and annotations. Nothing may be omitted or merged without explicit
   user approval.
3. **No font overlap**: Text labels must never overlap each other or overlap
   other visual elements. Ensure sufficient spacing between all text elements so
   every label is fully legible.
4. **Clean layout**: The redrawn image must have clean, well-organized
   formatting — tidy alignment, consistent spacing, uniform font sizes, and a
   clear visual hierarchy. The result should look publication-ready.
5. **File naming**: Before saving the redrawn image, rename the original file
   by appending an `_old` suffix before the extension (e.g.
   `framework.png` → `framework_old.png`). Then save the new image using the
   original filename. This ensures the old image is preserved for comparison
   while the new image is automatically used by the manuscript.
6. **Credentials**: Load fal.ai credentials from `C:\\source\\phdthesis\\.env`.
   Never expose secret values in manuscript text, code comments, logs, or saved
   artifacts.

---

## Revision Priorities

Unless the revision guidance explicitly suggests another order, prioritize in this sequence:

1. unify problem formulation and terminology
2. fix method clarity and notation consistency
3. strengthen experiments
4. improve figure or table quality and interpretation

---

## Progress Tracking and Update Rules

After each modification task is completed, update `docs/progress.md` with the following discipline:

0. **Progress file structure**:
   - Maintain three synchronized sections in `docs/progress.md`: `# 已全部修改`, `# 未修改或部分修改`, and `# 遗留问题`.
   - Use `# 遗留问题` for issues that still require the author's decision, confirmation, missing data, external resource, or other newly provided information.

1. **Recording completions**:
   - Add each completed task to the `# 已全部修改` section with a clear task number, brief description, and detailed modification notes.
   - Document what was changed and why it improves the paper.

2. **Moving completed items**:
   - If a task was previously listed in `# 未修改或部分修改`, remove it from that section and add it to `# 已全部修改` after completion.
   - Never leave completed tasks in the "未修改或部分修改" section.

3. **Progress markers**:
   - Mark in-progress tasks with `【进行中】`.
   - Mark blocked tasks with `【已阻挡】` followed by a concise reason and next steps.

4. **Inline user questions for incomplete items**:
   - For every item that remains under `# 未修改或部分修改`, add a short sub-list directly below that item describing what still requires the user's decision, answer, approval, or data.
   - Write these prompts so the user can respond directly inside `docs/progress.md` without needing a separate chat message.
   - If no user input is needed for a still-open item, state that no user decision is currently required and continue autonomous work.

5. **Legacy issue synchronization**:
   - If an incomplete or blocked item still requires author input, also record it in `# 遗留问题`.
   - In `# 遗留问题`, keep the question block explicit so the user can reply inline, for example with `需要你提供/决策：` followed by numbered prompts and `A:` answers.
   - When the issue is resolved and its corresponding work is moved into `# 已全部修改`, remove the matching entry from `# 遗留问题` in the same update.
   - If a new blocker, unanswered question, or missing dataset appears during execution, add a new entry to `# 遗留问题` immediately.

6. **Consistency across agents**:
   - All three agent rule files (Claude, OpenAI Codex, Copilot) must follow the same progress-tracking conventions.
   - When rule changes are made, update all three files to maintain consistency.

7. **Workflow closure**:
   - Continue iterating until all actionable items in `docs/revision_suggestions.tex` are either completed or explicitly marked as blocked.
   - A revision cycle is complete only when no unmarked actionable items remain.
