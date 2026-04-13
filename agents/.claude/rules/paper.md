# Claude Code Rules - Paper Revision

This paper-facing process may run in parallel with other agents and may repeat
when a broader revision workflow benefits from concurrent review, writing,
and figure generation.

The process should audit the current manuscript and only commit changes when
it is confident the edits improve publication quality and stay aligned with
`docs/revision_suggestions.tex` or with a fresh independent review when that
mode is active.

## Focus Areas

The paper-facing agent should focus on:
- fixing clarity, structure, and scientific narrative in the manuscript
- improving the argument and the contribution story
- inserting or updating figures/tables and captions
- ensuring the manuscript follows the target venue scope and formatting
- keeping the paper self-contained and reproducible

## Manuscript Independence Rule

The manuscript must stay independent from local repository organization.

Never write local implementation references into the paper, including:
- local folders such as `src/experiments/` or `src/figures/`
- script names such as `src/run_all.py`
- concrete local code filenames
- explanations centered on where code is stored

Instead, describe only reader-relevant scientific content: method design,
motivation, significance, experimental setup, metrics, results, and the
conclusions supported by those results.

## Multi-Agent Coordination

This agent may use other agents for:
- figures and table generation
- experiments or code enhancements in `src/`
- reference and citation updates
- consistency checks across `paper/` and `src/`

When using multiple agents, coordinate clearly and keep edits in separate
files or well-defined sections. Prefer batching related changes in one pass,
then re-checking the whole paper.

## Experiment Integration

If the revision guidance tells you to run experiments or add new results:
1. Write code under `src/`
2. Execute or simulate experiments if feasible
3. Integrate final results and plots into the manuscript
4. Rewrite any implementation-facing details into standalone scientific prose before they appear in the manuscript

## Blocked Items

If the current guidance is not actionable, switch to a fresh independent review
protocol and rewrite `docs/revision_suggestions.tex` with a new venue-driven
English-only review.
