"""Prompt corpus for trace capture: the mixed-regime serving mix plus LongBench prompts.

The mixed-regime generator is loaded from the original discovery script by
extracting its two pure functions with ``ast`` (so the corpus is identical to
the one used earlier, without importing that script's heavy dependencies).
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from src.experiments.tpds.common import REPO_ROOT

MIXED_REGIME_SCRIPT = REPO_ROOT / "src" / "experiments" / "run_mixed_regime_discovery.py"
LONGBENCH_ROOT = REPO_ROOT / "src" / "data" / "longbench_push120"

# Prompt templates of the earlier LongBench study.
LONGBENCH_TEMPLATES = {
    "narrativeqa": "Context:\n{context}\n\nQuestion: {input}\nAnswer:",
    "qasper": "Document:\n{context}\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en": "Document:\n{context}\n\nQuestion: {input}\nShort answer:",
    "gov_report": "Report:\n{context}\n\nWrite a concise summary:\n",
}


def _load_mixed_regime_generator():
    src = MIXED_REGIME_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {"_repeat_to_words", "_prompt_records"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise RuntimeError(f"could not find {wanted} in {MIXED_REGIME_SCRIPT}")
    module = ast.Module(body=nodes, type_ignores=[])
    ns: dict[str, Any] = {"random": random, "Any": Any}
    exec(compile(module, str(MIXED_REGIME_SCRIPT), "exec"), ns)
    return ns["_prompt_records"]


# Most mixed-regime prompts are 1-4K tokens long. To cover the 32..1024 range as
# well, the i-th sample of each regime is truncated to MIXED_LENGTH_CYCLE[i % 6]
# tokens (None = full prompt); short prompts are unaffected.
MIXED_LENGTH_CYCLE = (None, 64, 128, 256, 512, 1024)


def mixed_regime_prompts(samples_per_regime: int = 20, seed: int = 0, length_cycle=MIXED_LENGTH_CYCLE):
    gen = _load_mixed_regime_generator()
    out = []
    for r in gen(samples_per_regime, seed):
        i = int(r["id"].rsplit(":", 1)[1])
        out.append({"id": "mix_" + r["id"].replace(":", "_"), "regime": r["regime"], "source": "mixed_regime",
                    "text": r["prompt"], "max_tokens": length_cycle[i % len(length_cycle)]})
    return out


def longbench_prompts(per_task: int = 6, target_lengths=(1024, 2048, 4096, 8192),
                      root: Path = LONGBENCH_ROOT, tasks=None) -> list[dict[str, Any]]:
    tasks = tasks or list(LONGBENCH_TEMPLATES)
    out = []
    for task in tasks:
        path = Path(root) / task / "test.jsonl"
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i >= per_task:
                    break
                row = json.loads(line)
                text = LONGBENCH_TEMPLATES[task].format(context=row.get("context", ""), input=row.get("input", ""))
                out.append({"id": f"lb_{task}_{i:03d}", "regime": f"longbench_{task}", "source": "longbench",
                            "text": text, "max_tokens": int(target_lengths[i % len(target_lengths)])})
    return out


def build_corpus(samples_per_regime: int, longbench_per_task: int, target_lengths, seed: int = 0):
    return mixed_regime_prompts(samples_per_regime, seed) + longbench_prompts(longbench_per_task, target_lengths)


def prompt_fold(prompt_id: str, folds: int = 5) -> int:
    """Deterministic cross-validation fold of a prompt (used to form batches without leakage)."""
    return int(hashlib.sha1(prompt_id.encode("utf-8")).hexdigest(), 16) % folds
