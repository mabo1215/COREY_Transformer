---
name: progress-sync
description: Sync and report paper-revision progress across machines through git. Use when the user says "同步进度", "拉取最新进度", "换主机继续", "回顾实验进度", "给我实验进度表", "下一步计划实验的表", or invokes /progress-sync. Pulls the latest ledger, pushed-back results and paper submodule, shows review-item and experiment tables (purpose, host, % done, NZ-time ETA), and publishes local progress so another machine can resume.
---

# Progress sync across machines

Git is the only transport. The shared state:

| Path | Content |
|---|---|
| `docs/state/cycle.json` | current review cycle, venue, phase |
| `docs/state/items.json` | every review item and its status |
| `docs/state/experiments/<ID>.json` | one file per experiment: purpose, host, status, %, ETA, results, notes |
| `docs/state/hosts.json` | GPU host registry (no secrets) |
| `docs/state/lease.json` | which machine is currently driving the loop |
| `docs/progress.md` | human narrative + auto block rendered from the ledger |
| `src/experiments/results/` | result files pushed back by remote hosts |
| `paper/` | manuscript submodule |

## Modes

Default is **pull + report**. Say **push** to publish local work, **takeover** to take the
lease from another machine.

### pull + report

1. `bash src/revloop/sync.sh pull`
   - Exit 2 = conflict. For ledger JSON, keep per record the side with the newer
     `updated_at` and keep both sides' `notes`, then `git rebase --continue` and rerun.
   - `UNMERGED_RESULT_BRANCH origin/<b>` → merge it (`git merge --no-edit origin/<b>`), push,
     then delete the branch on origin.
2. `python src/revloop/revstate.py status` for warnings, then `revstate.py report` for the tables.
3. If a running experiment was last updated more than 2 h ago and the user wants fresh
   numbers, patrol it through the `remote-gpu` skill first.
4. Reply in Chinese with:
   - **实验进度表**: every experiment this cycle (ID, 实验, 意义, 主机, 状态, 进度,
     预计完成（新西兰时间）, 备注).
   - **下一步计划实验表**: only running and unfinished experiments, with the same columns
     plus estimated VRAM and hours. Finished experiments do not belong here.
   - Open review items and `# 遗留问题` that wait on the user.
   - Which machine holds the lease, and the "需要关注" warnings.

### push

```bash
python src/revloop/revstate.py render
bash src/revloop/sync.sh push "sync: <one-line summary>"
```

If the repository has `docs/ExperimentProgress.tex`, refresh its tables first. The
`paper-revise` skill's "Record" step says how.

`push` commits `paper/` first, then records its pointer and the other changes in the main
repository. It skips files over 20 MB and retries on non-fast-forward.

### takeover

Use it when the lease holder is another machine that is no longer working, for example
when you moved to another computer. Confirm with the user, then:

```bash
python src/revloop/revstate.py lease acquire --force --note "takeover"
bash src/revloop/sync.sh push "revloop: lease takeover"
```

## First use on a new machine

1. `git clone --recurse-submodules <repository URL>`
   (or run `git submodule update --init` in an existing clone).
2. Copy the rule template that is not tracked in the paper repository (`USAGE.md`,
   `.claude/rules/`, `agents/`) from the Bodhi_agent template or from an existing machine.
   The skills in `.claude/skills/` and the tooling in `src/revloop/` come with the clone.
3. In WSL, add `~/.ssh/config` aliases for the hosts in `docs/state/hosts.json`, using
   key-based login. The `remote-gpu` skill describes this; credentials are in
   `C:\source\.env`.
4. Optional: `setx REVLOOP_HOST <friendly-name>` on Windows (or `export` in WSL) so ledger
   entries name the machine readably.
5. `bash src/revloop/sync.sh pull`, then `python src/revloop/revstate.py status`.
