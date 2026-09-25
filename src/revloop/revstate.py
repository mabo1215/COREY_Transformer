#!/usr/bin/env python3
"""Shared revision-loop ledger stored under docs/state/.

Every machine that works on the paper -- local workstations and remote GPU
servers alike -- reads and writes the same JSON files, and git carries them
between machines. Only the standard library is used, so the script runs
unchanged on Windows, inside WSL, and on the remote Linux hosts.

Layout:
    docs/state/cycle.json             current review cycle (source, venue, phase)
    docs/state/items.json             review items of the current cycle
    docs/state/experiments/<ID>.json  one file per experiment (few merge conflicts)
    docs/state/hosts.json             remote host registry (never holds secrets)
    docs/state/lease.json             which machine currently drives the loop

Run `python src/revloop/revstate.py <command> --help` for details.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import sys
from pathlib import Path

def _find_root() -> Path:
    """Repository root: $REVLOOP_ROOT, else the nearest parent holding .git."""
    env = os.environ.get("REVLOOP_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


ROOT = _find_root()
STATE = ROOT / "docs" / "state"
EXP_DIR = STATE / "experiments"
CYCLE = STATE / "cycle.json"
ITEMS = STATE / "items.json"
HOSTS = STATE / "hosts.json"
LEASE = STATE / "lease.json"
PROGRESS = ROOT / "docs" / "progress.md"
USAGE = ROOT / "USAGE.md"
# Review files used by the repositories built from the rule template.
REVIEW_CANDIDATES = ("docs/revision_suggestions.tex", "docs/RevisionSuggestions.tex")

ITEM_STATUSES = ("todo", "in_progress", "waiting_exp", "done", "verified",
                 "blocked", "needs_author", "dropped")
ITEM_OPEN = {"todo", "in_progress", "waiting_exp"}
EXP_STATUSES = ("planned", "code_ready", "smoke_ok", "prepped", "queued",
                "running", "finished", "failed", "pulled", "integrated",
                "cancelled")
EXP_CLOSED = {"integrated", "cancelled"}
EXP_REMOTE_ACTIVE = {"queued", "running"}
LEASE_HOURS = 6.0
STALE_RUNNING_HOURS = 2.0
MAX_NOTES = 50

BEGIN = "<!-- revloop:status:begin -->"
END = "<!-- revloop:status:end -->"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
UTC = dt.timezone.utc


# --------------------------------------------------------------------------
# Time helpers (storage is UTC; display is New Zealand local time)
# --------------------------------------------------------------------------
def now() -> dt.datetime:
    return dt.datetime.now(UTC).replace(microsecond=0)


def iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text):
    if not text:
        return None
    try:
        t = dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=UTC)


def _load_nz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("Pacific/Auckland")
    except Exception:  # missing zoneinfo or tz database
        return None


NZ = _load_nz()


def _nz_offset_fallback(t_utc: dt.datetime) -> dt.timedelta:
    """NZ daylight time: last Sunday of September 02:00 NZST until the first
    Sunday of April 03:00 NZDT; both switches happen at 14:00 UTC on the
    preceding Saturday."""
    year = t_utc.year
    sep30 = dt.date(year, 9, 30)
    last_sun_sep = sep30 - dt.timedelta(days=(sep30.weekday() + 1) % 7)
    apr1 = dt.date(year, 4, 1)
    first_sun_apr = apr1 + dt.timedelta(days=(6 - apr1.weekday()) % 7)
    start = dt.datetime.combine(last_sun_sep, dt.time(14), UTC) - dt.timedelta(days=1)
    end = dt.datetime.combine(first_sun_apr, dt.time(14), UTC) - dt.timedelta(days=1)
    dst = t_utc >= start or t_utc < end
    return dt.timedelta(hours=13 if dst else 12)


def to_nz(t: dt.datetime) -> dt.datetime:
    if NZ is not None:
        return t.astimezone(NZ)
    off = _nz_offset_fallback(t.astimezone(UTC))
    return t.astimezone(dt.timezone(off))


def nz_str(value, short: bool = False) -> str:
    t = parse_iso(value) if isinstance(value, str) else value
    if t is None:
        return "-"
    z = to_nz(t)
    label = "NZDT" if z.utcoffset() == dt.timedelta(hours=13) else "NZST"
    return z.strftime("%m-%d %H:%M" if short else "%Y-%m-%d %H:%M") + " " + label


def age_hours(value) -> float | None:
    t = parse_iso(value)
    return None if t is None else (now() - t).total_seconds() / 3600.0


REL_RE = re.compile(r"^\+(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?$")


def parse_eta(text: str) -> str:
    """Accept '+3h30m', 'YYYY-MM-DD HH:MM' (NZ local time) or ISO-8601."""
    raw = text.strip()
    m = REL_RE.match(raw)
    if m and any(m.groups()):
        d, h, mi = (int(g) if g else 0 for g in m.groups())
        return iso(now() + dt.timedelta(days=d, hours=h, minutes=mi))
    cleaned = re.sub(r"\s*(NZT|NZST|NZDT)$", "", raw, flags=re.I)
    try:
        t = dt.datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
    except ValueError:
        sys.exit(f"cannot parse ETA {text!r}: use +3h30m, 'YYYY-MM-DD HH:MM' "
                 "(NZ local time) or ISO-8601 with an offset")
    if t.tzinfo is None:
        if NZ is not None:
            t = t.replace(tzinfo=NZ)
        else:
            guess = t.replace(tzinfo=UTC) - dt.timedelta(hours=12)
            t = (t - _nz_offset_fallback(guess)).replace(tzinfo=UTC)
    return iso(t)


# --------------------------------------------------------------------------
# Storage helpers
# --------------------------------------------------------------------------
def host_id() -> str:
    return os.environ.get("REVLOOP_HOST") or platform.node() or "unknown"


def load(path: Path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


def stamp(rec: dict) -> dict:
    rec["updated_at"] = iso(now())
    rec["updated_by"] = host_id()
    return rec


def add_note(rec: dict, text: str | None) -> None:
    if text:
        notes = rec.setdefault("notes", [])
        notes.append({"at": iso(now()), "by": host_id(), "text": text})
        del notes[:-MAX_NOTES]


def last_note(rec: dict) -> str:
    notes = rec.get("notes") or []
    return notes[-1]["text"] if notes else ""


def csv_list(text: str | None) -> list[str]:
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def check_id(value: str) -> str:
    if not ID_RE.match(value):
        sys.exit(f"invalid id {value!r}: use letters, digits, '.', '_' or '-'")
    return value


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items() -> dict:
    return load(ITEMS, {"items": {}})["items"]


def save_items(items: dict) -> None:
    save(ITEMS, {"items": items})


def exp_path(exp_id: str) -> Path:
    return EXP_DIR / f"{exp_id}.json"


def load_exp(exp_id: str) -> dict | None:
    return load(exp_path(exp_id), None)


def all_exps() -> list[dict]:
    if not EXP_DIR.exists():
        return []
    return [load(p, {}) for p in sorted(EXP_DIR.glob("*.json"))]


def load_hosts() -> dict:
    return load(HOSTS, {"hosts": {}})["hosts"]


def venue_from_usage() -> str | None:
    if not USAGE.exists():
        return None
    match = re.search(r"目标会议或期刊[:：]\s*(.+)", USAGE.read_text(encoding="utf-8"))
    if not match:
        return None
    venue = match.group(1).strip().strip("<>").strip()
    return venue or None


def print_field(value) -> None:
    if isinstance(value, list):
        for entry in value:
            print(entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False))
    elif isinstance(value, dict):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    elif value is not None:
        print(value)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_init(args) -> int:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if not ITEMS.exists():
        save_items({})
    if not HOSTS.exists():
        save(HOSTS, {"hosts": {}})
    if not LEASE.exists():
        save(LEASE, {"host": None})
    print(f"ledger ready at {rel(STATE)}")
    return 0


def cmd_cycle(args) -> int:
    cyc = load(CYCLE, {})
    if args.action == "show":
        print(json.dumps(cyc, indent=2, ensure_ascii=False))
        return 0
    if args.action == "start":
        source = args.source or next((c for c in REVIEW_CANDIDATES if (ROOT / c).exists()), None)
        if source is None:
            sys.exit(f"no review file found ({' or '.join(REVIEW_CANDIDATES)}); pass --source")
        src = ROOT / source
        if not src.exists():
            sys.exit(f"review source not found: {source}")
        items = load_items()
        open_items = [i for i, r in items.items() if r.get("status") in ITEM_OPEN]
        if cyc.get("cycle_id") and items:
            if open_items and not args.archive:
                sys.exit(f"cycle {cyc['cycle_id']} still has {len(open_items)} open items; "
                         "pass --archive to archive it and start a new one")
            dest = STATE / "archive" / cyc["cycle_id"]
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ITEMS, dest / "items.json")
            shutil.copy2(CYCLE, dest / "cycle.json")
            print(f"archived previous cycle to {rel(dest)}")
        started = now()
        cyc = {
            "cycle_id": args.id or f"{started:%Y%m%d}-{src.stem}",
            "source": rel(src),
            "source_sha256": sha256(src),
            "venue": args.venue or venue_from_usage() or cyc.get("venue"),
            "phase": "revise",
            "started_at": iso(started),
        }
        save(CYCLE, stamp(cyc))
        save_items({})
        print(f"cycle {cyc['cycle_id']} started from {cyc['source']} (venue: {cyc['venue']})")
        return 0
    if not cyc.get("cycle_id"):
        sys.exit("no active cycle; run 'cycle start' first")
    if args.action == "set":
        if args.phase:
            cyc["phase"] = args.phase
        if args.venue:
            cyc["venue"] = args.venue
    elif args.action == "rehash":
        cyc["source_sha256"] = sha256(ROOT / cyc["source"])
    save(CYCLE, stamp(cyc))
    print(json.dumps(cyc, indent=2, ensure_ascii=False))
    return 0


def cmd_item(args) -> int:
    items = load_items()
    if args.action == "list":
        wanted = set(csv_list(args.status))
        for item_id, rec in sorted(items.items()):
            if wanted and rec.get("status") not in wanted:
                continue
            exps = ",".join(rec.get("exps") or []) or "-"
            print(f"{item_id}\t{rec.get('status')}\t{rec.get('kind', '-')}\t{exps}\t{rec.get('title', '')}")
        return 0
    item_id = check_id(args.id)
    if args.action == "show":
        rec = items.get(item_id)
        if rec is None:
            sys.exit(f"unknown item {item_id}")
        print_field(rec.get(args.field) if args.field else rec)
        return 0
    rec = items.get(item_id) or {"id": item_id, "status": "todo", "exps": [],
                                 "notes": [], "created_at": iso(now())}
    for field in ("title", "kind", "where", "question"):
        value = getattr(args, field)
        if value is not None:
            rec[field] = value
    if args.status:
        rec["status"] = args.status
    if args.exp is not None:
        rec["exps"] = csv_list(args.exp)
    for exp_id in csv_list(args.add_exp):
        if exp_id not in rec["exps"]:
            rec["exps"].append(exp_id)
    add_note(rec, args.note)
    items[item_id] = stamp(rec)
    save_items(items)
    print(f"item {item_id}: {rec['status']}")
    return 0


def cmd_exp(args) -> int:
    if args.action == "list":
        wanted = set(csv_list(args.status))
        for rec in all_exps():
            if wanted and rec.get("status") not in wanted:
                continue
            if args.host and rec.get("host") != args.host:
                continue
            if args.format == "tsv":
                log = rec.get("log") or f"src/logs/revloop/{rec['id']}.log"
                print("\t".join([rec["id"], rec.get("status", ""), rec.get("screen") or "-", log]))
            else:
                print(f"{rec['id']}\t{rec.get('status')}\t{rec.get('host') or '-'}\t"
                      f"{rec.get('progress', 0)}%\t{nz_str(rec.get('eta'), short=True)}\t{rec.get('title', '')}")
        return 0
    exp_id = check_id(args.id)
    rec = load_exp(exp_id)
    if args.action == "show":
        if rec is None:
            sys.exit(f"unknown experiment {exp_id}")
        print_field(rec.get(args.field) if args.field else rec)
        return 0
    if rec is None:
        rec = {"id": exp_id, "status": "planned", "progress": 0, "items": [],
               "results": [], "notes": [], "created_at": iso(now())}
    for field in ("title", "purpose", "host", "screen", "log", "cmd", "design_ref"):
        value = getattr(args, field)
        if value is not None:
            rec[field] = value
    if args.items is not None:
        rec["items"] = csv_list(args.items)
    if args.vram is not None:
        rec["vram_gb"] = args.vram
    if args.est_hours is not None:
        rec["est_hours"] = args.est_hours
    if args.results_clear:
        rec["results"] = []
    for pattern in args.result or []:
        if pattern not in rec["results"]:
            rec["results"].append(pattern)
    if args.progress is not None:
        rec["progress"] = max(0, min(100, int(args.progress)))
    if args.eta is not None:
        rec["eta"] = None if args.eta.lower() in ("", "none", "-") else parse_eta(args.eta)
    if args.status:
        rec["status"] = args.status
        if args.status == "running" and not rec.get("started_at"):
            rec["started_at"] = iso(now())
        if args.status in ("finished", "failed"):
            rec["finished_at"] = iso(now())
        if args.status in ("finished", "pulled", "integrated"):
            rec["progress"] = 100
            rec["eta"] = None
    add_note(rec, args.note)
    save(exp_path(exp_id), stamp(rec))
    print(f"experiment {exp_id}: {rec['status']} {rec.get('progress', 0)}%")
    return 0


def cmd_host(args) -> int:
    hosts = load_hosts()
    if args.action == "list":
        for name, rec in sorted(hosts.items()):
            print(f"{name}\t{rec.get('label', '')}\tssh={rec.get('ssh_alias') or '?'}\t"
                  f"vram={rec.get('vram_gb') or '?'}GB\tchannel={rec.get('result_channel', 'git')}\t"
                  f"state={rec.get('state', 'unknown')}")
        return 0
    rec = hosts.get(args.name)
    if args.action == "show":
        if rec is None:
            sys.exit(f"unknown host {args.name}; known: {', '.join(sorted(hosts)) or 'none'}")
        print_field(rec.get(args.field) if args.field else rec)
        return 0
    rec = rec or {}
    for pair in args.pairs:
        if "=" not in pair:
            sys.exit(f"expected key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        if value.lower() in ("null", "none", ""):
            rec[key] = None
        elif value.lower() in ("true", "false"):
            rec[key] = value.lower() == "true"
        elif re.fullmatch(r"-?\d+(\.\d+)?", value):
            rec[key] = float(value) if "." in value else int(value)
        else:
            rec[key] = value
    hosts[args.name] = stamp(rec)
    save(HOSTS, {"hosts": hosts})
    print(f"host {args.name} updated")
    return 0


def cmd_lease(args) -> int:
    lease = load(LEASE, {})
    me, t = host_id(), now()
    holder = lease.get("host")
    expires = parse_iso(lease.get("expires_at"))
    active = bool(holder) and expires is not None and expires > t
    if args.action == "show":
        if active:
            print(f"held by {holder} until {nz_str(expires)} ({lease.get('note') or 'no note'})")
        else:
            print("free" + (f" (last holder {holder}, expired)" if holder else ""))
        return 0
    if args.action == "acquire":
        if active and holder != me and not args.force:
            print(f"LEASE HELD by {holder} until {nz_str(expires)} "
                  f"({lease.get('note') or 'no note'}); rerun with --force to take over")
            return 3
        keep_start = active and holder == me
        lease = {
            "host": me,
            "acquired_at": lease.get("acquired_at") if keep_start else iso(t),
            "expires_at": iso(t + dt.timedelta(hours=args.hours)),
            "note": args.note or (lease.get("note") if keep_start else None),
        }
        if active and holder != me:
            lease["taken_over_from"] = holder
        save(LEASE, lease)
        print(f"lease held by {me} until {nz_str(lease['expires_at'])}")
        return 0
    if active and holder != me and not args.force:
        print(f"lease belongs to {holder}; not released (use --force)")
        return 3
    save(LEASE, {"host": None, "released_at": iso(t), "released_by": me})
    print("lease released")
    return 0


def cmd_reconcile(args) -> int:
    """Mark experiments as pulled once all their declared result files exist
    locally and are newer than the run start."""
    changed = 0
    for rec in all_exps():
        if rec.get("status") not in ("queued", "running", "finished") or not rec.get("results"):
            continue
        started = parse_iso(rec.get("started_at"))
        complete = True
        for pattern in rec["results"]:
            matches = [Path(p) for p in glob.glob(str(ROOT / pattern))]
            if not matches:
                complete = False
                break
            if started and not any(dt.datetime.fromtimestamp(p.stat().st_mtime, UTC) >= started
                                   for p in matches):
                complete = False
                break
        if complete:
            rec["status"] = "pulled"
            rec["progress"] = 100
            rec["eta"] = None
            add_note(rec, "auto: all declared result files are present locally")
            save(exp_path(rec["id"]), stamp(rec))
            print(f"PULLED {rec['id']}")
            changed += 1
    if not changed:
        print("reconcile: no newly arrived results")
    return 0


def cmd_results(args) -> int:
    """Print existing files matched by the experiments' declared result globs,
    one repo-relative path per line (used to force-add them to git even when
    they live under an ignored directory such as outputs/)."""
    wanted_ids = set(csv_list(args.exp))
    wanted_status = set(csv_list(args.status))
    seen = set()
    for rec in all_exps():
        if wanted_ids and rec["id"] not in wanted_ids:
            continue
        if wanted_status and rec.get("status") not in wanted_status:
            continue
        for pattern in rec.get("results") or []:
            for match in sorted(glob.glob(str(ROOT / pattern), recursive=True)):
                path = Path(match)
                if path.is_file() and path not in seen:
                    seen.add(path)
                    print(rel(path))
    return 0


def attention() -> list[str]:
    notes = []
    cyc = load(CYCLE, {})
    if cyc.get("source"):
        src = ROOT / cyc["source"]
        if not src.exists():
            notes.append(f"review source {cyc['source']} is missing")
        elif sha256(src) != cyc.get("source_sha256"):
            notes.append("review source changed since the cycle started -> re-sync items "
                         "('cycle rehash' for small edits, 'cycle start --archive' for a new round)")
    exps = {r["id"]: r for r in all_exps()}
    for rec in exps.values():
        status, hours = rec.get("status"), age_hours(rec.get("updated_at"))
        if status in EXP_REMOTE_ACTIVE and hours is not None and hours > STALE_RUNNING_HOURS:
            notes.append(f"{rec['id']} is {status} on {rec.get('host') or '?'} but was last "
                         f"updated {hours:.1f}h ago -> patrol")
        if status == "finished":
            notes.append(f"{rec['id']} finished remotely but results are not local yet -> pull")
        if status == "pulled":
            notes.append(f"{rec['id']} results are local but not integrated -> backfill the paper")
        if status == "failed":
            notes.append(f"{rec['id']} failed -> diagnose, fix and relaunch, or block its item")
        eta = parse_iso(rec.get("eta"))
        if status == "running" and eta and eta < now():
            notes.append(f"{rec['id']} is past its ETA ({nz_str(eta, short=True)}) -> patrol")
    for item_id, rec in sorted(load_items().items()):
        linked = [exps[e] for e in rec.get("exps") or [] if e in exps]
        missing = [e for e in rec.get("exps") or [] if e not in exps]
        if missing:
            notes.append(f"item {item_id} references unknown experiments {', '.join(missing)}")
        if rec.get("status") == "waiting_exp" and linked and all(
                e.get("status") in EXP_CLOSED | {"failed"} for e in linked):
            notes.append(f"item {item_id} is waiting_exp but all its experiments are closed -> re-evaluate")
    return notes


def cmd_status(args) -> int:
    cyc = load(CYCLE, {})
    items = load_items()
    exps = all_exps()
    lease = load(LEASE, {})
    if args.json:
        print(json.dumps({"cycle": cyc, "items": items, "experiments": exps,
                          "lease": lease, "attention": attention(),
                          "host": host_id()}, indent=2, ensure_ascii=False))
        return 0
    print(f"revloop status @ {nz_str(now())} (this host: {host_id()})")
    if cyc.get("cycle_id"):
        print(f"cycle: {cyc['cycle_id']}  phase={cyc.get('phase')}  venue={cyc.get('venue')}  "
              f"source={cyc.get('source')}")
    else:
        print("cycle: none -> run 'cycle start' (review file is auto-detected)")
    expires = parse_iso(lease.get("expires_at"))
    if lease.get("host") and expires and expires > now():
        print(f"lease: {lease['host']} until {nz_str(expires)}")
    else:
        print("lease: free")
    counts: dict[str, int] = {}
    for rec in items.values():
        counts[rec.get("status", "?")] = counts.get(rec.get("status", "?"), 0) + 1
    summary = ", ".join(f"{s} {counts[s]}" for s in ITEM_STATUSES if s in counts)
    print(f"items: {summary or 'none'} (total {len(items)})")
    for item_id, rec in sorted(items.items()):
        if rec.get("status") in ITEM_OPEN | {"blocked", "needs_author"}:
            exps_txt = ",".join(rec.get("exps") or []) or "-"
            print(f"  {item_id:<10} {rec.get('status'):<12} exps={exps_txt:<14} {rec.get('title', '')}")
    open_exps = [e for e in exps if e.get("status") not in EXP_CLOSED]
    print(f"experiments: {len(open_exps)} open / {len(exps)} total")
    for rec in open_exps:
        hours = age_hours(rec.get("updated_at"))
        age = f"{hours:.1f}h ago" if hours is not None else "?"
        print(f"  {rec['id']:<10} {rec.get('status'):<10} {rec.get('host') or '-':<9} "
              f"{rec.get('progress', 0):>3}%  eta {nz_str(rec.get('eta'), short=True):<17} "
              f"upd {age:<9} {rec.get('title', '')}")
    notes = attention()
    if notes:
        print("attention:")
        for note in notes:
            print(f"  - {note}")
    return 0


# --------------------------------------------------------------------------
# Rendering (docs/progress.md block and chat report)
# --------------------------------------------------------------------------
ITEM_LABEL = {"todo": "待办", "in_progress": "进行中", "waiting_exp": "等待实验",
              "done": "已完成", "verified": "已复核", "blocked": "已阻挡",
              "needs_author": "等待作者", "dropped": "不处理"}
EXP_LABEL = {"planned": "已规划", "code_ready": "代码就绪", "smoke_ok": "冒烟通过",
             "prepped": "远端就绪", "queued": "排队中", "running": "运行中",
             "finished": "已完成(未拉回)", "failed": "失败", "pulled": "已拉回",
             "integrated": "已回填论文", "cancelled": "已取消"}


def cell(text) -> str:
    return str(text if text not in (None, "") else "-").replace("|", "\\|").replace("\n", " ")


def build_report() -> str:
    cyc = load(CYCLE, {})
    items = load_items()
    exps = all_exps()
    lease = load(LEASE, {})
    lines = []
    head = f"更新时间：{nz_str(now())} · 主机：{host_id()}"
    if cyc.get("cycle_id"):
        head += (f" · 周期：{cyc['cycle_id']} · 阶段：{cyc.get('phase')} · "
                 f"审稿来源：`{cyc.get('source')}` · 目标：{cyc.get('venue') or '-'}")
    else:
        head += " · 尚未启动修订周期"
    expires = parse_iso(lease.get("expires_at"))
    if lease.get("host") and expires and expires > now():
        head += f" · 当前执行主机：{lease['host']}（至 {nz_str(expires, short=True)}）"
    lines += [head, ""]

    lines += ["### 审稿条目", ""]
    if items:
        lines += ["| ID | 条目 | 类型 | 状态 | 关联实验 | 最新备注 |", "|---|---|---|---|---|---|"]
        for item_id, rec in sorted(items.items()):
            lines.append("| " + " | ".join(cell(v) for v in (
                item_id, rec.get("title"), rec.get("kind"),
                ITEM_LABEL.get(rec.get("status"), rec.get("status")),
                ", ".join(rec.get("exps") or []), last_note(rec))) + " |")
    else:
        lines.append("（暂无登记条目）")
    lines.append("")

    lines += ["### 实验进度表", ""]
    if exps:
        lines += ["| ID | 实验 | 意义 | 主机 | 状态 | 进度 | 预计完成（新西兰时间） | 备注 |",
                  "|---|---|---|---|---|---|---|---|"]
        for rec in exps:
            lines.append("| " + " | ".join(cell(v) for v in (
                rec["id"], rec.get("title"), rec.get("purpose"), rec.get("host"),
                EXP_LABEL.get(rec.get("status"), rec.get("status")),
                f"{rec.get('progress', 0)}%", nz_str(rec.get("eta")), last_note(rec))) + " |")
    else:
        lines.append("（暂无登记实验）")
    lines.append("")

    pending = [e for e in exps if e.get("status") not in EXP_CLOSED | {"pulled"}]
    lines += ["### 下一步计划（进行中与未完成的实验）", ""]
    if pending:
        lines += ["| ID | 实验 | 意义 | 主机 | 状态 | 进度 | 预计完成（新西兰时间） | 预估显存/时长 |",
                  "|---|---|---|---|---|---|---|---|"]
        for rec in pending:
            budget = f"{rec.get('vram_gb', '?')} GB / {rec.get('est_hours', '?')} h"
            lines.append("| " + " | ".join(cell(v) for v in (
                rec["id"], rec.get("title"), rec.get("purpose"), rec.get("host"),
                EXP_LABEL.get(rec.get("status"), rec.get("status")),
                f"{rec.get('progress', 0)}%", nz_str(rec.get("eta")), budget)) + " |")
    else:
        lines.append("（无进行中或未完成的实验）")
    notes = attention()
    if notes:
        lines += ["", "### 需要关注", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines).rstrip() + "\n"


def cmd_report(args) -> int:
    sys.stdout.write(build_report())
    return 0


def cmd_render(args) -> int:
    target = ROOT / args.file
    block = (f"{BEGIN}\n"
             "<!-- Auto-generated by `python src/revloop/revstate.py render`; edit the ledger in docs/state/, not this block. -->\n"
             "## 自动同步状态（revloop）\n\n"
             f"{build_report()}{END}")
    crlf = False
    if target.exists():
        with target.open(encoding="utf-8", newline="") as handle:
            raw = handle.read()
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
    else:
        text = "# 论文进度\n\n# 已全部修改\n\n# 未修改或部分修改\n\n# 遗留问题\n"
    if BEGIN in text and END in text:
        before, rest = text.split(BEGIN, 1)
        after = rest.split(END, 1)[1]
        text = before + block + after
    else:
        lines = text.splitlines(keepends=True)
        idx = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
        if idx is None:
            text = block + "\n\n" + text
        else:
            text = "".join(lines[:idx + 1]) + "\n" + block + "\n\n" + "".join(lines[idx + 1:]).lstrip("\n")
    with target.open("w", encoding="utf-8", newline="\r\n" if crlf else "\n") as handle:
        handle.write(text)
    print(f"rendered status block into {rel(target)}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the docs/state/ skeleton").set_defaults(func=cmd_init)

    p = sub.add_parser("status", help="summary with attention list")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    sub.add_parser("report", help="print Markdown progress tables").set_defaults(func=cmd_report)

    p = sub.add_parser("render", help="refresh the auto block in docs/progress.md")
    p.add_argument("--file", default="docs/progress.md")
    p.set_defaults(func=cmd_render)

    sub.add_parser("reconcile", help="mark experiments whose results arrived as pulled").set_defaults(func=cmd_reconcile)

    p = sub.add_parser("results", help="list existing files matched by declared result globs")
    p.add_argument("--exp", help="comma-separated experiment ids (default: all)")
    p.add_argument("--status", help="comma-separated status filter")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("cycle", help="review cycle: start | set | rehash | show")
    p.add_argument("action", choices=("start", "set", "rehash", "show"))
    p.add_argument("--source", help="review file (default: the first of "
                   + ", ".join(REVIEW_CANDIDATES) + " that exists)")
    p.add_argument("--id")
    p.add_argument("--venue")
    p.add_argument("--phase", choices=("revise", "verify", "gate", "done"))
    p.add_argument("--archive", action="store_true", help="archive an unfinished cycle")
    p.set_defaults(func=cmd_cycle)

    p = sub.add_parser("item", help="review items: set | show | list")
    p.add_argument("action", choices=("set", "show", "list"))
    p.add_argument("id", nargs="?")
    p.add_argument("--title")
    p.add_argument("--kind", help="text | experiment | figure | citation | format | other")
    p.add_argument("--status", choices=ITEM_STATUSES)
    p.add_argument("--exp", help="comma-separated experiment ids (replaces the list)")
    p.add_argument("--add-exp", help="comma-separated experiment ids to append")
    p.add_argument("--where", help="manuscript location, e.g. 'Sec 6, Table 2'")
    p.add_argument("--question", help="open question for the author (blocked / needs_author)")
    p.add_argument("--note")
    p.add_argument("--field", help="show: print one field only")
    p.set_defaults(func=cmd_item)

    p = sub.add_parser("exp", help="experiments: set | show | list")
    p.add_argument("action", choices=("set", "show", "list"))
    p.add_argument("id", nargs="?")
    p.add_argument("--title")
    p.add_argument("--purpose", help="why the experiment matters for the review")
    p.add_argument("--items", help="comma-separated review item ids")
    p.add_argument("--host", help="registry name from docs/state/hosts.json, or 'local'")
    p.add_argument("--status", help=f"one of {', '.join(EXP_STATUSES)} (list: comma-separated filter)")
    p.add_argument("--progress", type=int)
    p.add_argument("--eta", help="+3h30m | 'YYYY-MM-DD HH:MM' (NZ time) | ISO-8601 | none")
    p.add_argument("--screen")
    p.add_argument("--log", help="remote log path relative to the repo root")
    p.add_argument("--cmd", help="command or lane script that runs the experiment")
    p.add_argument("--design-ref", dest="design_ref", help="e.g. 'docs/Design.md#E3'")
    p.add_argument("--result", action="append", help="result path or glob (repeatable, appended)")
    p.add_argument("--results-clear", action="store_true")
    p.add_argument("--vram", type=float, help="estimated peak VRAM in GB")
    p.add_argument("--est-hours", dest="est_hours", type=float)
    p.add_argument("--note")
    p.add_argument("--field", help="show: print one field only")
    p.add_argument("--format", choices=("text", "tsv"), default="text", help="list output format")
    p.set_defaults(func=cmd_exp)

    p = sub.add_parser("host", help="remote host registry: list | show | set")
    p.add_argument("action", choices=("list", "show", "set"))
    p.add_argument("name", nargs="?")
    p.add_argument("pairs", nargs="*", help="set: key=value pairs")
    p.add_argument("--field")
    p.set_defaults(func=cmd_host)

    p = sub.add_parser("lease", help="which machine drives the loop: acquire | release | show")
    p.add_argument("action", choices=("acquire", "release", "show"))
    p.add_argument("--hours", type=float, default=LEASE_HOURS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--note")
    p.set_defaults(func=cmd_lease)
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.command == "exp" and args.action == "set" and args.status and args.status not in EXP_STATUSES:
        sys.exit(f"unknown experiment status {args.status!r}; use one of {', '.join(EXP_STATUSES)}")
    if args.command in ("item", "exp") and args.action in ("set", "show") and not args.id:
        sys.exit(f"{args.command} {args.action} needs an id")
    if args.command == "host" and args.action in ("show", "set") and not args.name:
        sys.exit(f"host {args.action} needs a host name")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
