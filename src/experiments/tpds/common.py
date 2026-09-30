"""Shared utilities for the TPDS experiments.

Provides: environment metadata, resumable JSONL/CSV writers (flush + fsync per
row, skip completed keys on restart), disk-space guards, CUDA-event timing
with confidence intervals, and a tee logger into ``src/logs/tpds/``.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "src" / "experiments" / "results" / "tpds"
SMOKE_DIR = RESULTS_DIR / "smoke"
LOG_DIR = REPO_ROOT / "src" / "logs" / "tpds"
# Captured activation traces are large and must not be committed; outputs/ is git-ignored.
DEFAULT_TRACE_DIR = REPO_ROOT / "src" / "outputs" / "tpds_traces"

MAMBA1_CHUNKS = (32, 64, 128, 256, 512, 1024)
MAMBA2_CHUNKS = (32, 64, 128, 256, 512)
GDN_CHUNKS = (16, 32, 64)


def host_name() -> str:
    return os.environ.get("REVLOOP_HOST") or socket.gethostname()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- logging
class Logger:
    """Print to stdout and append to a log file under src/logs/tpds/."""

    def __init__(self, name: str, log_dir: Path | None = None):
        self.dir = Path(log_dir or LOG_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{name}.log"
        self._fh = open(self.path, "a", encoding="utf-8")

    def __call__(self, *parts: Any) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] " + " ".join(str(p) for p in parts)
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())


# --------------------------------------------------------------------------- env metadata
def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def env_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "host": host_name(),
        "hostname": socket.gethostname(),
        "utc": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            meta["gpu"] = p.name
            meta["gpu_mem_gb"] = round(p.total_memory / 2**30, 2)
            meta["gpu_sm_count"] = p.multi_processor_count
            meta["gpu_cc"] = f"{p.major}.{p.minor}"
    except Exception as exc:  # pragma: no cover
        meta["torch_error"] = repr(exc)
    for pkg in ("triton", "transformers", "mamba_ssm", "fla-core", "flash-linear-attention", "numpy", "scikit-learn"):
        meta[f"pkg_{pkg}"] = _pkg_version(pkg)
    meta["nvidia_smi"] = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"])
    try:
        meta["git_commit"] = _cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"])
    except Exception:
        pass
    return meta


# --------------------------------------------------------------------------- disk guard
def check_disk(path: Path, need_gb: float, need_inodes: int = 1000, log: Callable[..., None] = print) -> None:
    """Fail fast when the target filesystem lacks byte or inode capacity."""
    path = Path(path)
    probe = path
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / 2**30
    try:
        st = os.statvfs(probe)
        free_inodes = st.f_favail if st.f_files > 0 else None
    except Exception:
        free_inodes = None
    log(f"disk check {probe}: free {free_gb:.1f} GB (need {need_gb:.1f}), free inodes {free_inodes}")
    if free_gb < need_gb:
        raise SystemExit(f"Not enough disk space at {probe}: {free_gb:.1f} GB free < {need_gb:.1f} GB needed")
    if free_inodes is not None and free_inodes < need_inodes:
        raise SystemExit(f"Not enough inodes at {probe}: {free_inodes} < {need_inodes}")


# --------------------------------------------------------------------------- resumable writers
def _fsync(fh) -> None:
    fh.flush()
    os.fsync(fh.fileno())


class JsonlWriter:
    """Append-only JSONL writer; one flushed and fsynced row per call.

    Rows carry a ``key`` field; keys already present in the file are reported by
    ``done`` so the caller can skip them on restart. A header row with
    ``{"kind": "meta"}`` is appended on every (re)start.
    """

    def __init__(self, path: Path, meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # truncated last line from an interrupted write
                    if row.get("kind") != "meta" and "key" in row:
                        self.done.add(row["key"])
        self._fh = open(self.path, "a", encoding="utf-8")
        if meta is not None:
            self.write({"kind": "meta", **meta}, track=False)

    def write(self, row: dict[str, Any], track: bool = True) -> None:
        self._fh.write(json.dumps(row, default=_json_default) + "\n")
        _fsync(self._fh)
        if track and "key" in row:
            self.done.add(row["key"])

    def close(self) -> None:
        self._fh.close()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def read_jsonl(path: Path, kind: str | None = None) -> list[dict[str, Any]]:
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kind is None:
                if row.get("kind") != "meta":
                    rows.append(row)
            elif row.get("kind") == kind:
                rows.append(row)
    return rows


def dedupe_by_key(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last row per key (rows may be duplicated if a run was re-forced)."""
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        out[r.get("key", id(r))] = r
    return list(out.values())


class CsvWriter:
    """Resumable CSV writer with a fixed column list; fsync after each row."""

    def __init__(self, path: Path, columns: list[str], key_col: str = "key"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = columns
        self.done: set[str] = set()
        new = not self.path.exists() or self.path.stat().st_size == 0
        if not new:
            with open(self.path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                existing_cols = reader.fieldnames or []
                if existing_cols != columns:
                    raise SystemExit(f"{self.path} has different columns; refuse to append (move it aside first)")
                for row in reader:
                    if row.get(key_col):
                        self.done.add(row[key_col])
        self._fh = open(self.path, "a", encoding="utf-8", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=columns, extrasaction="ignore")
        if new:
            self._w.writeheader()
            _fsync(self._fh)
        self.key_col = key_col

    def write(self, row: dict[str, Any]) -> None:
        self._w.writerow(row)
        _fsync(self._fh)
        self.done.add(str(row[self.key_col]))

    def close(self) -> None:
        self._fh.close()


def write_json(path: Path, obj: Any) -> None:
    """Atomic JSON write (temporary file, fsync, rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
        _fsync(fh)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- statistics
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_crit(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    keys = sorted(_T975)
    for k in keys:
        if dof <= k:
            return _T975[k]
    return 1.96


def summarize(samples: list[float]) -> dict[str, float]:
    n = len(samples)
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if n > 1 else 0.0
    half = t_crit(n - 1) * std / math.sqrt(n) if n > 1 else float("nan")
    s = sorted(samples)
    return {
        "n": n,
        "mean_ms": mean,
        "std_ms": std,
        "ci95_lo_ms": mean - half,
        "ci95_hi_ms": mean + half,
        "median_ms": s[n // 2],
        "min_ms": s[0],
    }


# --------------------------------------------------------------------------- timing
_FLUSH_BUF = None


def l2_flush() -> None:
    """Overwrite a buffer larger than any current L2 cache (128 MB)."""
    import torch

    global _FLUSH_BUF
    if _FLUSH_BUF is None:
        _FLUSH_BUF = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device="cuda")
    _FLUSH_BUF.zero_()


_SPIN_CYCLES_PER_MS: float | None = None


def _spin_cycles_per_ms() -> float:
    """GPU clock cycles per millisecond of ``torch.cuda._sleep``, measured once."""
    import torch

    global _SPIN_CYCLES_PER_MS
    if _SPIN_CYCLES_PER_MS is None:
        cycles = 10_000_000
        torch.cuda._sleep(cycles)          # warm up
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.cuda._sleep(cycles)
        e.record()
        torch.cuda.synchronize()
        _SPIN_CYCLES_PER_MS = cycles / max(s.elapsed_time(e), 1e-3)
    return _SPIN_CYCLES_PER_MS


def cuda_time(fn: Callable[[], Any], warmup: int = 10, repeats: int = 50, flush_l2: bool = False,
              pre_spin_ms: float = 0.0) -> list[float]:
    """Time ``fn`` with CUDA events; returns per-repeat milliseconds.

    ``pre_spin_ms`` > 0 enqueues a GPU-side spin of that length before the timed repeats, so that
    the host can queue every (event, kernel, event) triple while the GPU is still busy. Host-side
    stalls (for example CPU load from other tenants) then cannot insert idle gaps between a start
    event and its kernel. With 0 (the default) the behaviour is unchanged.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    if pre_spin_ms > 0:
        torch.cuda._sleep(int(pre_spin_ms * _spin_cycles_per_ms()))
    for i in range(repeats):
        if flush_l2:
            l2_flush()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def peak_vram_gb() -> float:
    import torch

    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


_NVML_HANDLE: Any = None


_OWN_GPU_PIDS: set[int] = set()


def all_gpu_pids(index: int = 0) -> list[int] | None:
    """PIDs that nvidia-smi lists as compute processes on GPU ``index``; None if they cannot be listed."""
    out = _cmd(["nvidia-smi", f"--id={index}", "--query-compute-apps=pid", "--format=csv,noheader"])
    if out is None:
        # nvidia-smi ran but printed nothing (no compute apps), or is unavailable
        return [] if shutil.which("nvidia-smi") else None
    return [int(t.strip().strip(",")) for t in out.split() if t.strip().strip(",").isdigit()]


def register_own_gpu_pids(baseline: list[int] | None, index: int = 0,
                          log: Callable[[str], None] | None = None) -> set[int]:
    """Record the PID(s) under which nvidia-smi lists this process.

    Inside a container nvidia-smi can report host PIDs, which differ from ``os.getpid()``.
    Call with the PID list taken *before* this process created its CUDA context; the PIDs
    that appeared since then are this process. Returns the registered set.
    """
    now = all_gpu_pids(index)
    if now is None or baseline is None:
        return set(_OWN_GPU_PIDS)
    new = set(now) - set(baseline) - {os.getpid()}
    _OWN_GPU_PIDS.update(new)
    if log:
        log(f"own GPU PIDs as listed by nvidia-smi: {sorted(new) or [os.getpid()]}")
    return set(_OWN_GPU_PIDS)


def foreign_gpu_pids(index: int = 0) -> list[int] | None:
    """PIDs of compute processes on GPU ``index`` other than this process; None if they cannot be listed."""
    pids = all_gpu_pids(index)
    if pids is None:
        return None
    return [p for p in pids if p != os.getpid() and p not in _OWN_GPU_PIDS]


def wait_for_exclusive_gpu(until_epoch: float | None, log: Callable[[str], None] | None = None,
                           poll_s: float = 30.0, index: int = 0) -> float | None:
    """Block while other processes hold the GPU.

    Returns the seconds spent waiting (0.0 if the GPU was free), or None if ``until_epoch``
    passes while the GPU is still busy.
    """
    announced = False
    t0 = time.time()
    while True:
        pids = foreign_gpu_pids(index)
        if not pids:  # empty list, or None when processes cannot be listed (guard then does nothing)
            if announced and log:
                log("GPU exclusive again; resuming")
            return time.time() - t0
        if until_epoch is not None and time.time() >= until_epoch:
            if log:
                log(f"GPU still busy (pids {pids}) at the guard deadline; stopping")
            return None
        if not announced and log:
            log(f"foreign GPU processes {pids}; pausing timings")
            announced = True
        time.sleep(poll_s)


def gpu_state(index: int = 0) -> dict[str, Any] | None:
    """SM/memory clock (MHz), power (W), temperature (C) and active clock-event (throttle) reasons.

    Uses NVML when ``pynvml`` is importable, else one ``nvidia-smi`` query. Returns None if neither works.
    """
    global _NVML_HANDLE
    try:
        import pynvml  # type: ignore

        if _NVML_HANDLE is None:
            pynvml.nvmlInit()
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(index)
        h = _NVML_HANDLE
        reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        return {"sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                "mem_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
                "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "throttle_reasons": int(reasons), "source": "nvml"}
    except Exception:
        pass
    out = _cmd(["nvidia-smi", f"--id={index}", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,"
                "clocks_event_reasons.active", "--format=csv,noheader,nounits"])
    if not out:
        return None
    try:
        sm, mem, pw, temp, reasons = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"sm_mhz": float(sm), "mem_mhz": float(mem), "power_w": float(pw), "temp_c": float(temp),
                "throttle_reasons": int(reasons, 16) if reasons.startswith("0x") else reasons, "source": "nvidia-smi"}
    except ValueError:
        return None


def entropy_rule_chunk(r: float, c_min: int = 32, c_max: int = 512) -> int:
    """Eq. (2) of the manuscript: map normalized entropy r in [0,1] to a power-of-two chunk."""
    r = min(max(float(r), 0.0), 1.0)
    c = 2 ** round(math.log2(c_min + r * (c_max - c_min)))
    return int(min(max(c, c_min), c_max))
