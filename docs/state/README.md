# Revision-loop ledger

Machine-readable state of the current revision cycle. Git carries it between machines:
the local workstations and the remote GPU hosts. Any machine can pull it and continue
the work. Edit it only through `python src/revloop/revstate.py`, which stamps every
change with a UTC time and the machine name. Humans read the tables that `render` writes
into `docs/progress.md`.

| File | Content |
|---|---|
| `cycle.json` | Review cycle: review file and its SHA-256, venue, phase (`revise` → `verify` → `gate` → `done`) |
| `items.json` | Review items: title, kind, status, linked experiments, manuscript location, notes |
| `experiments/<ID>.json` | One experiment per file: purpose, linked items, host, status, progress %, ETA, screen, log, result globs, notes |
| `hosts.json` | Remote GPU hosts: SSH alias, repo root, Python, git proxy, VRAM, result channel, shutdown command. Never passwords. |
| `lease.json` | The machine currently driving the loop, and when that claim expires |
| `archive/<cycle>/` | Items of earlier cycles |

Item statuses: `todo`, `in_progress`, `waiting_exp`, `done`, `verified`, `blocked`,
`needs_author`, `dropped`.

Experiment lifecycle:

```
planned → code_ready → smoke_ok → prepped → queued → running → finished → pulled → integrated
                                                             ↘ failed      (cancelled at any point)
```

What moves an experiment along:
- `remote_watch.sh` on the GPU host marks it `finished` or `failed` and pushes the result files.
- `revstate.py reconcile` (run by `sync.sh pull`) marks it `pulled` once the files are present locally.
- The revision loop marks it `integrated` after backfilling the paper.

Times are stored in UTC and shown in New Zealand time. JSON is written sorted, with LF
line endings, and with one file per experiment, so concurrent edits from different
machines rarely conflict. When they do, keep the record with the newer `updated_at` and
merge both sides' `notes`.
