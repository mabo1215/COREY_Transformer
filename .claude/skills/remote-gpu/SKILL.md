---
name: remote-gpu
description: Run this paper's experiments on remote GPU servers (H800, PRO 6000, vGPU 3090, 4c, ...) and record every step in the git-synced ledger docs/state/. Use when the user says "<主机> 无卡模式已开", "<主机> GPU已开" / "已开卡", "巡检远端实验", "拉回实验结果", "<主机> 关机", or invokes /remote-gpu; paper-revise also hands remote work to it. Covers no-GPU preparation, VRAM-maximising parallel launch in screen sessions, 10-minute patrols, pulling results back and safe shutdown.
---

# Remote GPU experiments

Hosts are registered in `docs/state/hosts.json` (`python src/revloop/revstate.py host list`).
The registry never contains secrets.

Login details (`.claude/rules/host-login.md`) are kept in `C:\source\.env`, in blocks of
three lines: host label, `ssh -p <port> root@<host>`, password. A host's `login_ref` can
name another file: `C:\work\9\saveway.txt` holds the PRO 6000 and H800 blocks (labels `pro 6000`, `H800`). Use them only to
set up SSH. Never copy a password into the repository, logs, commit messages, notes or
chat.

Every remote action goes through WSL (repository rule):

```bash
wsl bash src/revloop/remote.sh ping|sync|patrol <host>
wsl bash src/revloop/remote.sh exec <host> <script.sh> [args]    # any other remote work
wsl bash src/revloop/remote.sh launch <host> <screen> <lane-script> [args]
wsl bash src/revloop/remote.sh watch <host> [--shutdown] [--interval 600] <EXP_ID> ...
wsl bash src/revloop/remote.sh fetch <host> [EXP_ID ...]          # rsync channel only
wsl bash src/revloop/remote.sh shutdown <host> --yes [--force]
```

For anything beyond these subcommands, write a bash script into the scratchpad and run
it with `remote.sh exec`. Do not build multi-line remote commands in PowerShell. From Git
Bash, pass absolute `/mnt/...` paths only with `MSYS_NO_PATHCONV=1`; repo-relative paths
are fine.

## 0. Resolve host and mode

Map the user's wording to a registry name, case- and whitespace-insensitively: H800 →
`h800`, PRO 6000 → `pro6000`, vGPU 3090 → `vgpu3090`, 4c → `4c`. Register any other label
(A800, 5090, vGPU 4090, …) with `revstate.py host set <name> label="…" vram_gb=…`.

If `ssh_alias`, `root` or `python` is missing, or `remote.sh ping` fails with a
key/permission error, set up access once:
1. Find the host's block in `C:\source\.env` (or its `login_ref`); otherwise ask the user.
2. In WSL, create a key if none exists and add an `~/.ssh/config` alias with
   `HostName`, `Port`, `User` and `IdentityFile`.
3. Install the key.
   - With `sshpass`: write the password to a mode-600 temporary file, run
     `sshpass -f <file> ssh-copy-id -p <port> root@<host>`, and delete the file immediately.
   - Without `sshpass`: ask the user to run `ssh-copy-id` once and type the password themselves.
4. Record the host:
   `revstate.py host set <host> ssh_alias=… root=<repo path on host> python=… git_proxy="source /etc/network_turbo"`.
   Leave `git_proxy` out on hosts that are not AutoDL.
5. Confirm with `remote.sh ping <host>`.

SSH runs in BatchMode, so password prompts fail fast instead of hanging. Registry edits
are shared state: push them.

If the host cannot reach GitHub even with `network_turbo` (see `.claude/rules/shell.md`):
- set `result_channel=rsync`;
- copy code with `rsync`/`scp` through WSL instead of `git pull`;
- use `fetch` for results.

| The user says | Mode |
|---|---|
| 无卡模式已开 (no-GPU mode on) | `prep` |
| GPU已开 / 已开卡 / 开始实验 | `run` |
| 巡检 / 进度 | `patrol` |
| 拉回结果 | `pull` |
| 关机 | `shutdown` |

Before any mode, run `bash src/revloop/sync.sh pull`. Before `prep` or `run`, run
`bash src/revloop/sync.sh push "<msg>"` and then `remote.sh sync <host>`. The host must
have the current code **and** the ledger files of the experiments it will run.

## prep: no-GPU mode (code and data only, no GPU jobs)

1. `remote.sh ping` (expect "no GPU visible"). Then `remote.sh exec` a prep script that
   does the following:
   - checks the Python environment, model weights, datasets and free disk;
   - builds data with CPU-only steps;
   - runs `bash -n` on every lane script and `python -m py_compile` on changed files;
   - runs a CPU or synthetic smoke test where one exists.
2. Write the lane scripts for the GPU session now (see `run`), commit them under
   `src/experiments/remote/` (English), and sync the host again.
3. For each prepared experiment: `exp set <E> --status prepped --host <host> --note "prep ok: …"`.
   On failures, fix and retry, or record `--note` and a `# 遗留问题` entry if the user must act.
4. `revstate.py render`, `sync.sh push`, then tell the user (in Chinese) that everything is
   ready to power on. Include the lane plan: which experiments, estimated peak VRAM and hours.

## run: GPU on

1. `remote.sh sync <host>`, then `remote.sh ping` (GPUs must be visible).
2. **Pack lanes for maximum VRAM use.**
   - Budget ≈ 0.9 × host `vram_gb` (80 GB H800, 96 GB PRO 6000, 48 GB vGPU 3090).
   - Take each experiment's `vram_gb` from smoke tests or earlier logs. If it is unknown,
     start the job alone, read `nvidia-smi` after 2–3 min, record it with `exp set --vram`,
     then add the next lane.
   - Put long jobs first; run independent seeds and shards in parallel lanes.
   - Keep the summed peaks within budget.
3. **One lane per screen.** A lane is a script of wrapper calls, so every job leaves markers:
   ```bash
   bash src/revloop/remote_run.sh E5 -- python src/experiments/run_x.py --seed 42 …
   bash src/revloop/remote_run.sh E6 -- python src/experiments/run_y.py …
   ```
   Launch each lane: `remote.sh launch <host> <screen> src/experiments/remote/<lane>.sh`.
4. **Start the watcher**: `remote.sh watch <host> --shutdown <all EXP_IDs of this session>`.
   - Each experiment that finishes or fails is written to its ledger file, its result files
     are committed and pushed (to a fallback `revloop-results-*` branch if main rejects the
     push), and once all are done the host powers off.
   - `--shutdown` is the default for paid hosts: USAGE's GPU triggers say to shut down after
     the results come back. Omit it only if the user asked to keep the host on.
   - The watcher needs git push access on the host. Hosts without it get
     `result_channel=rsync`: no watcher, and the local patrol does `fetch` and then `shutdown`.
5. Ledger: `exp set <E> --status running --host <host> --screen <screen> --eta +<h>h`.
   Then `render` and `sync.sh push`.
6. After 2–3 min, patrol. If memory is well under budget and queued work remains, add a
   lane. On OOM, move a job to a later lane.
7. **Ten-minute patrols** (USAGE): if this session should keep watching, start
   `/loop 10m /remote-gpu patrol <host>` via the `loop` skill. The watcher already
   guarantees push and power-off, so the local loop only mirrors progress, catches
   failures early, and stops itself once every experiment on the host is closed.

## patrol

`wsl bash src/revloop/remote.sh patrol <host>` prints screens, GPU utilisation and memory,
disk, the watcher's log tail, and each experiment's markers and log tail. Then:

- Update each running experiment with `exp set <E> --progress <pct> --eta "<YYYY-MM-DD HH:MM>" --note "…"`.
  ETA is NZ local time; `--eta +2h` also works.
  - Derive % from the job's own counters (`[k/N]`, epoch, shard) and the ETA from its
    measured rate.
  - If there is no counter, say so in the note instead of guessing.
- `FAIL` marker → diagnose from the log.
  - Code bug: fix, smoke-test locally, push, `remote.sh sync`, relaunch that job, restart
    the watcher for it.
  - Otherwise: mark it `failed` with the reason.
- No progress across two patrols while the GPU is idle → investigate (dead screen, hung
  I/O, full disk).
- `render` + `sync.sh push` only when something changed (avoid commit noise).
- All experiments on the host closed and results in git → stop the patrol loop.

## pull

- **git channel:** `bash src/revloop/sync.sh pull`. Its `reconcile` flips experiments to
  `pulled` when their result files are present and newer than the run start. Merge any
  `UNMERGED_RESULT_BRANCH` (see paper-revise step 1).
- **rsync channel:** `wsl bash src/revloop/remote.sh fetch <host> [EXP_ID ...]`, then
  `sync.sh push "results: …"`.
- Before trusting a `pulled` experiment, check that its files are complete: the JSON
  parses and the expected splits and seeds are present. Integration into the paper happens
  in paper-revise.

## shutdown

Allowed only when all of the following hold:
- the results are persisted (pushed to git, or fetched and committed locally);
- no experiment on the host is `queued`, `running` or `finished`-but-not-pulled;
- the user's wording authorised power-off (the GPU triggers in USAGE do: "完成后…关机").

Then run `wsl bash src/revloop/remote.sh shutdown <host> --yes`, followed by `sync.sh push`
(the registry records `state=off`). On AutoDL, issue exactly one `shutdown -h now`; never
`shutdown -c` or a delayed shutdown.

## Safety

- No GPU jobs in no-GPU mode.
- Never delete remote datasets, caches or checkpoints unless the user asks.
- Never shut a host down with unpushed results if its `disk_persists` is not `true`.
- Keep large artefacts (`*.pt`, checkpoints, caches) out of git. An experiment's
  `--result` globs must match only small result files (JSON/CSV/figures). Those files
  are force-added even under ignored directories such as `outputs/`; any file over 20 MB
  is skipped.
- Remote commits identify themselves as `revloop-<host>`; do not change the host's global
  git config.
