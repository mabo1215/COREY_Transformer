# Claude Code Rules - Source Code and Experiments

## Venue-Driven Standards

When `USAGE.md` declares a target venue through the line
`目标会议或期刊：...`, use that venue as the canonical standard for scope,
formatting, and quality checks.

## Role and Responsibilities

This support role may run in parallel with other agents as part of a
broader paper revision process.

The source-facing agent should focus on:
- ensuring experiments are reproducible and properly documented
- keeping code structure consistent with the repository conventions
- preparing final figures for insertion into `paper/figs/`
- verifying that added code supports the paper claims
- cleaning up implementation details and comments

## Paper Integration

The source-facing agent should not modify the manuscript directly unless the
revision guidance explicitly asks it to update textual descriptions based on
new experimental results.

When providing material for the manuscript, convert implementation details into
paper-appropriate scientific descriptions. Do not ask the paper to mention
local folders, script names, or concrete code filenames from `src/`.

## Artifact Management

If new experiments are needed:
1. Implement them under `src/`
2. Save final analysis artifacts in `src/figures/` or `paper/figs/` as appropriate
3. Hand off only reader-facing summaries of setup, metrics, results, and conclusions to manuscript-facing rules

## Language Standards

Keep all code, comments, docstrings, and logs in English.
