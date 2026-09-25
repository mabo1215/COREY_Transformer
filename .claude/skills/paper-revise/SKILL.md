---
name: paper-revise
description: Resumable, multi-machine paper revision loop. Use when the user says "继续当前 revision cycle", "论文循环修订", "按审稿意见循环修改论文", "消费遗留问题回答并继续推进", "巡检运行中任务并继续推进", or invokes /paper-revise. Works through the review file (docs/revision_suggestions.tex or docs/RevisionSuggestions.tex) item by item (manuscript edits, experiment design, local smoke tests, remote GPU runs, result backfill), records every step in the git-synced ledger docs/state/, and keeps iterating until every review item is verified or explicitly blocked.
---

# Paper revision loop

One keyword drives the whole revision cycle. All state lives in git — the ledger
in `docs/state/`, `docs/progress.md`, result files under `src/experiments/results/`
and the `paper/` submodule — so any machine can pull and continue where another
one stopped, including after remote GPU results are pushed back.

**Review file** below means `docs/revision_suggestions.tex`, or
`docs/RevisionSuggestions.tex` in repositories that use that name. `cycle start`
detects which one exists.

Tools, run from the repository root:

| Tool | Purpose |
|---|---|
| `python src/revloop/revstate.py …` | ledger: `status`, `report`, `render`, `cycle`, `item`, `exp`, `host`, `lease`, `reconcile` (`--help` on each) |
| `bash src/revloop/sync.sh pull` / `push "<msg>"` / `status` | git sync of repo + `paper/` submodule; `pull` also reconciles arrived results |
| `remote-gpu` skill | everything that touches a remote GPU host |
| `progress-sync` skill | cross-machine status and report tables |

On Windows use `python` (never the `python3` Store stub) and Git Bash for
`sync.sh`. Remote work is WSL-first: `wsl bash src/revloop/remote.sh …` with
repo-relative paths (from Git Bash, absolute `/mnt/...` paths need `MSYS_NO_PATHCONV=1`).

## First use in a repository

If `src/revloop/` is missing, bootstrap once:
1. Copy `src/revloop/` from the rule template (`C:\source\Bodhi_agent\src\revloop\`)
   or from another repository that already has it.
2. Make git track the skills and the ledger.
   - `.gitignore` must not hide `.claude/skills/` or `docs/state/`. If it ignores
     `.claude/`, replace that line with `.claude/*` and `!.claude/skills/`.
     Add `src/logs/revloop/`.
   - `.gitattributes` needs `src/revloop/** text eol=lf`, `docs/state/** text eol=lf` and
     `.claude/skills/** text eol=lf`. Otherwise Windows clones with `core.autocrlf=true`
     check the scripts out with CRLF and bash fails.
   - Result files may live under an ignored directory such as `outputs/`: the tools
     force-add the files an experiment declares in `--result` (up to 20 MB each).
3. `python src/revloop/revstate.py init`. Register the GPU hosts with
   `revstate.py host set …`, as the `remote-gpu` skill describes.
4. Commit and push, so every machine and GPU host gets the tooling.

## Ground rules

- Read `USAGE.md` and `.claude/rules/*.md` when present. A paper repository may not
  track them in git, so a fresh machine can lack them; `docs/state/cycle.json` keeps
  the venue. Target venue = the `目标会议或期刊：` line.
- English only in `paper/`, `src/`, the review file, `docs/Design.md`;
  `docs/progress.md` may be Chinese.
- Never write repository paths, script names or code file names into `paper/`.
- Never rewrite the review file in this loop. A new review round goes
  through the Stage 9 gate in `USAGE.md`: ask in `# 遗留问题`, proceed only on `A: 是`.
- Nothing is `done` without evidence: a diff, a build log, or a result file.
- Only ask the user when a decision changes scientific direction or needs data,
  credentials, money (powering on GPUs) or authorship judgement. Otherwise act.

## One iteration

### 1. Sync and claim

```bash
bash src/revloop/sync.sh pull
python src/revloop/revstate.py lease acquire --note "paper-revise"
bash src/revloop/sync.sh push "revloop: lease"
python src/revloop/revstate.py status
```

- `pull` exit 2: rebase conflict. For `docs/state/**/*.json`, merge by hand and keep, per
  record, the side with the newer `updated_at` while keeping both sides' `notes`; for other
  files resolve normally. Then `git rebase --continue` and rerun.
- `UNMERGED_RESULT_BRANCH origin/<b>`: a remote pushed results to a fallback branch.
  `git merge --no-edit origin/<b>` (ledger conflicts as above), push, then
  `git push origin --delete <b>` once the merge is on origin.
- `lease acquire` exit 3: another machine is driving the loop. Show holder and expiry and
  ask the user whether to take over (`--force`). Never continue silently.

### 2. Make sure the cycle matches the review

- No cycle → `revstate.py cycle start` (auto-detects the review file), read the
  whole review, and register every actionable item using the review's own numbering:
  `revstate.py item set W3 --title "…" --kind text|experiment|figure|citation|format --where "Sec 6, Table 2"`.
- `status` says the review source changed → small edit: `cycle rehash` and adjust items;
  new round: `cycle start --archive`.
- Newer real reviewer feedback in `docs/` than the review file (for example an
  official review export) → Stage 9 gate question in `# 遗留问题`; keep working on the
  current review until the author answers.

### 3. Consume author answers

In `docs/progress.md` → `# 遗留问题`, every `A: …` line is an author decision. Apply it to
the matching item (`item set … --status … --note "author: …"`) and act on it. Remove a
question block only when its item is completed.

### 4. Reconcile experiments

- `sync.sh pull` already ran `reconcile`: experiments whose declared result files arrived
  are now `pulled`.
- `queued`/`running` experiments: patrol through the `remote-gpu` skill (progress %, NZ ETA).
  If a host is off or unreachable, record that; never invent progress.
- `failed`: read the log tail, fix, smoke-test locally, relaunch — or block the item with
  the reason.

### 5. Do the work

Take open work in this order. Parallelise independent pieces with subagents that own
disjoint files, then merge and re-check.

1. **Integrate `pulled` results.** Update numbers, tables and figures in `paper/` from the
   result files, add the analysis prose, then `exp set <E> --status integrated`. Mark the
   item `done` once nothing else is pending for it.
2. **Text items**, in the repository's priority: problem formulation and terminology →
   method clarity and notation → evaluation narrative → everything else.
3. **Experiment items without experiments:**
   1. Plan in `docs/Design.md`: purpose, review item, setup, metrics, runtime, peak VRAM, host.
   2. Register: `exp set E5 --title "…" --purpose "<why it answers the review>" --items W3 --host h800 --vram 30 --est-hours 2 --result "src/experiments/results/<name>*.json" --design-ref "docs/Design.md#E5"`,
      then `item set W3 --status waiting_exp --add-exp E5`.
   3. Implement under `src/`, then smoke-test on the local GPU. Use the local Python
      environment that has CUDA-enabled torch. On success: `exp set E5 --status smoke_ok`.
   4. `sync.sh push` (code **and** ledger must be on origin before any remote step), then
      hand over to `remote-gpu`: `prep` if the user said the host is in no-GPU mode, `run`
      if the GPU is on. If no host is available, add a `# 遗留问题` entry such as
      "请开启 H800 无卡模式" and continue with other items.
4. **Figures, citations, formatting.** Figure redraws follow the fal.ai rules in `USAGE.md`;
   new references must be real and verified.

After each edit batch, build the paper (`paper\build.bat` via cmd when present, else the
repository's standard LaTeX build) and check the logs: no errors, no `??`, no overfull
boxes that show, page limit respected.

### 6. Record after every batch

```bash
python src/revloop/revstate.py item set W3 --status done --note "Table 2 + Sec 6 updated; build clean"
python src/revloop/revstate.py render
bash src/revloop/sync.sh push "revise: W3 unified binding-accuracy definition"
```

Also keep the narrative sections of `docs/progress.md` in line with the repository rules:
`# 已全部修改`, `# 未修改或部分修改` (with inline questions under each open item),
`# 遗留问题`. `render` owns only the block between `<!-- revloop:status:begin -->` and
`<!-- revloop:status:end -->`; never edit inside it by hand.

If the repository has `docs/ExperimentProgress.tex`, update it from `revstate.py report`
in the same batch:
- **First table** (Item | Status | Progress | Required outcome or decision | ETA): one row
  per review item, with completion %.
- **Next-step plan**: only running and not-yet-run experiments, with %, NZ-time ETA and
  notes. Remove experiments that have finished.
- Update the snapshot date.
- Keep the file English and compilable.

### 7. Decide what comes next

- **Local work left** → next iteration from step 4. Re-run `sync.sh pull` only after waiting
  or when another machine may have pushed.
- **Only remote work left** (experiments running, or waiting for the user to power a host on):
  - make sure each running host has its watcher (`remote-gpu` `run` mode starts it), so
    results are pushed and the host powers off even if no local session is open;
  - `lease release` + `sync.sh push`;
  - inside a `/loop` run, schedule the next wake-up (about 10 min while GPUs run, 30 min
    while waiting on the user); otherwise tell the user how to resume: type the keyword
    again later on any machine, or run `/loop 10m /paper-revise`.
- **Every item `done`, `blocked`, `needs_author` or `dropped`, and no open experiments** →
  verify:
  1. **Stage 9' re-check.** Spawn a fresh subagent with no prior context. It reads
     the review file and the current paper (`.tex` plus freshly built
     `.pdf`) and reports, per item, whether it is addressed, partly addressed or not
     addressed, with evidence. Anything short of fully addressed goes back to `todo`
     with the finding as a note, and the loop continues.
  2. **Stage 11 gate.** Clean build; venue page limit and template; no `??` or undefined
     references; no repository paths in the paper; every reported number traceable to a
     result file; every citation present in `references.bib`; anonymity if the venue is
     double-blind.
  3. **Pass** → `item set <id> --status verified` for each item, `cycle set --phase done`,
     `render`, `lease release`, then `sync.sh push`. Do not start a new review round on
     your own.

## Blocked items

Use `blocked` for an external limit and `needs_author` for a decision or missing data.
Either status needs a reason (`--note`), the open question (`--question`), and a matching
block in `# 遗留问题` with numbered prompts and empty `A:` lines. Keep working on everything
else.

## End of every turn

Reply in Chinese with a short summary:
- what changed (items and experiments);
- what is running on which host, with its NZ-time ETA (from `revstate.py report`);
- what the user needs to do (power a host on, answer `遗留问题`);
- confirmation that everything was pushed, and which machine holds the lease, if any.
