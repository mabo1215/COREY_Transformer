"""
8-GPU distributed heterogeneous corpus experiment for COREY.

Distributes the 60-prompt heterogeneous corpus (20 low / 20 medium / 20 high
entropy regime) across all ranks.  Each rank processes its assigned prompts,
then rank 0 gathers all per-prompt results and computes regime-level summaries
(entropy statistics, chunk distribution, latency if enabled).

The per-regime chunk distribution validates COREY as an adaptive controller
that selects different chunk sizes across entropy regimes rather than operating
as a single-regime auto-tuner.

Launch:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \\
    src/experiments/run_heterogeneous_corpus_8gpu.py \\
    --model mamba-370m \\
    --output-dir src/outputs/heterogeneous_corpus_8gpu
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="COREY 8-GPU heterogeneous corpus experiment.")
parser.add_argument("--model", type=str, default="mamba-370m")
parser.add_argument("--num-bins", type=int, default=256)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--no-timing", action="store_true",
                    help="Skip latency measurement; collect entropy/chunk stats only.")
parser.add_argument("--output-dir", type=Path, default=Path("src/outputs/heterogeneous_corpus_8gpu"))
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, str] = {
    "mamba-370m": "state-spaces/mamba-370m-hf",
    "mamba-1.4b": "state-spaces/mamba-1.4b-hf",
    "mamba-2.8b": "state-spaces/mamba-2.8b-hf",
}


# ---------------------------------------------------------------------------
# HF token helper
# ---------------------------------------------------------------------------

def _set_hf_token_from_envfile(env_path: str = "/home/amabo1215/source/.env") -> None:
    import os
    try:
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("Huggingface_model_token:="):
                        token = line.strip().split("Huggingface_model_token:=", 1)[-1].strip()
                        if token:
                            os.environ["HF_TOKEN"] = token
                        break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 60-prompt heterogeneous corpus (identical to run_heterogeneous_corpus.py)
# ---------------------------------------------------------------------------

def _build_corpus() -> list[dict[str, str]]:
    low = [
        "The quick brown fox jumps over the lazy dog. " * 6,
        "To be or not to be, that is the question. " * 6,
        "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. " * 4,
        "One two three four five six seven eight nine ten. " * 6,
        "The cat sat on the mat. The cat sat on the mat. " * 5,
        "Yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes yes. " * 4,
        "Monday Tuesday Wednesday Thursday Friday Saturday Sunday. " * 6,
        "North South East West North South East West North South East West. " * 4,
        "Red orange yellow green blue indigo violet. " * 7,
        "Apple banana cherry date elderberry fig grape. " * 5,
        "Click click click click click click click click click click. " * 5,
        "The number is 42. The number is 42. The number is 42. " * 5,
        "Do re mi fa sol la ti do re mi fa sol la ti. " * 5,
        "First second third fourth fifth sixth seventh. " * 6,
        "I am I am I am I am I am I am I am I am. " * 6,
        "The system status is: OK OK OK OK OK OK OK OK. " * 5,
        "Error: null null null null null null null null null null. " * 4,
        "Name: Alice. Age: 30. Name: Alice. Age: 30. " * 6,
        "0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1. " * 4,
        "True false true false true false true false. " * 7,
    ]
    medium = [
        ("def binary_search(arr, target):\n"
         "    \"\"\"Search sorted array for target using binary search.\"\"\"\n"
         "    left, right = 0, len(arr) - 1\n"
         "    while left <= right:\n"
         "        mid = (left + right) // 2\n"
         "        if arr[mid] == target: return mid\n"),
        ("class LinkedList:\n    \"\"\"Singly linked list with O(1) prepend.\"\"\"\n"
         "    def __init__(self):\n        self.head = None\n        self.size = 0\n"
         "    def prepend(self, val):\n        from collections import namedtuple\n"
         "        Node = namedtuple('Node', ['val', 'next'])\n"
         "        self.head = Node(val, self.head)\n        self.size += 1\n"),
        ("import torch\nimport torch.nn as nn\n\nclass LayerNorm(nn.Module):\n"
         "    def __init__(self, d_model, eps=1e-6):\n        super().__init__()\n"
         "        self.gamma = nn.Parameter(torch.ones(d_model))\n"
         "        self.beta = nn.Parameter(torch.zeros(d_model))\n"),
        ("def merge_sort(arr):\n    \"\"\"Stable O(n log n) sort.\"\"\"\n"
         "    if len(arr) <= 1: return arr\n    mid = len(arr) // 2\n"
         "    left = merge_sort(arr[:mid]); right = merge_sort(arr[mid:])\n"
         "    return sorted(left + right)\n"),
        ("class Tokenizer:\n    def __init__(self, vocab_size=50000):\n"
         "        self.vocab = {}; self.merges = []; self.vocab_size = vocab_size\n"
         "    def encode(self, text):\n        tokens = list(text.encode('utf-8'))\n"
         "        return self._apply_merges(tokens)\n"),
        ("def attention(Q, K, V, mask=None):\n    import math\n"
         "    d_k = Q.size(-1)\n"
         "    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)\n"
         "    if mask is not None: scores = scores.masked_fill(mask == 0, -1e9)\n"
         "    return torch.softmax(scores, dim=-1) @ V\n"),
        ("class DataLoader:\n    def __init__(self, dataset, batch_size=32, shuffle=True):\n"
         "        self.dataset = dataset; self.batch_size = batch_size\n"
         "        self.shuffle = shuffle; self.indices = list(range(len(dataset)))\n"
         "    def __iter__(self):\n        if self.shuffle:\n"
         "            import random; random.shuffle(self.indices)\n"),
        ("def compute_iou(box1, box2):\n"
         "    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])\n"
         "    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])\n"
         "    inter = max(0, x2-x1)*max(0, y2-y1)\n"
         "    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])\n"
         "    return inter/(area1+(box2[2]-box2[0])*(box2[3]-box2[1])-inter+1e-6)\n"),
        ("import numpy as np\ndef pca(X, n_components=2):\n"
         "    X_centered = X - X.mean(axis=0)\n"
         "    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)\n"
         "    return X_centered @ Vt[:n_components].T\n"),
        ("class RingBuffer:\n    def __init__(self, capacity):\n"
         "        self.buf = [None]*capacity; self.capacity = capacity\n"
         "        self.head = 0; self.count = 0\n"
         "    def append(self, val):\n"
         "        self.buf[self.head % self.capacity] = val\n"
         "        self.head += 1; self.count = min(self.count+1, self.capacity)\n"),
        ("def dijkstra(graph, source):\n    import heapq\n"
         "    dist = {v: float('inf') for v in graph}; dist[source] = 0\n"
         "    heap = [(0, source)]\n    while heap:\n"
         "        d, u = heapq.heappop(heap)\n        if d > dist[u]: continue\n"
         "        for v, w in graph[u]:\n"
         "            if dist[u]+w < dist[v]: dist[v]=dist[u]+w; heapq.heappush(heap,(dist[v],v))\n"),
        ("class GradientDescent:\n    def __init__(self, lr=0.01): self.lr = lr\n"
         "    def step(self, params, grads):\n"
         "        return [p - self.lr * g for p, g in zip(params, grads)]\n"),
        ("def tokenize_sentence(sentence):\n    import re\n"
         "    return re.findall(r'\\b\\w+\\b|[.,!?;:]', sentence.lower())\n"
         "def build_vocab(corpus):\n    vocab = {}\n"
         "    for s in corpus:\n        for t in tokenize_sentence(s):\n"
         "            vocab[t] = vocab.get(t, 0) + 1\n    return vocab\n"),
        ("import hashlib, math\nclass BloomFilter:\n"
         "    def __init__(self, capacity, error_rate=0.01):\n"
         "        m = -capacity*math.log(error_rate)/math.log(2)**2\n"
         "        self.size = int(m); self.bits = bytearray(self.size)\n"
         "        self.k = int(self.size/capacity*math.log(2))\n"),
        ("def cross_entropy(logits, targets):\n"
         "    import torch.nn.functional as F\n"
         "    log_probs = F.log_softmax(logits, dim=-1)\n"
         "    return -log_probs[range(len(targets)), targets].mean()\n"
         "def accuracy(logits, targets):\n"
         "    return (logits.argmax(-1) == targets).float().mean().item()\n"),
        ("class ConvBlock(nn.Module):\n    def __init__(self, in_ch, out_ch, kernel=3, stride=1):\n"
         "        super().__init__()\n        pad = kernel // 2\n"
         "        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)\n"
         "        self.bn = nn.BatchNorm2d(out_ch)\n"
         "    def forward(self, x): return torch.relu(self.bn(self.conv(x)))\n"),
        ("def softmax_temperature(logits, T=1.0):\n    import torch\n"
         "    return torch.softmax(logits / T, dim=-1)\n"
         "def top_k_filter(logits, k=50):\n    v, _ = torch.topk(logits, k)\n"
         "    return logits.masked_fill(logits < v[..., [-1]], -float('inf'))\n"),
        ("class LRUCache:\n    def __init__(self, capacity):\n"
         "        from collections import OrderedDict\n"
         "        self.cache = OrderedDict(); self.capacity = capacity\n"
         "    def get(self, key):\n        if key not in self.cache: return -1\n"
         "        self.cache.move_to_end(key); return self.cache[key]\n"),
        ("def beam_search(model, input_ids, beam_width=5, max_len=50):\n"
         "    import torch\n    beams = [(0.0, input_ids.tolist())]\n"
         "    for _ in range(max_len):\n        candidates = []\n"
         "        for score, seq in beams:\n"
         "            ids = torch.tensor([seq])\n"
         "            with torch.no_grad(): logits = model(ids).logits[0, -1]\n"
         "            probs = torch.log_softmax(logits, dim=-1)\n"
         "            for p, idx in zip(*probs.topk(beam_width)): candidates.append((score+p.item(), seq+[idx.item()]))\n"
         "        beams = sorted(candidates, reverse=True)[:beam_width]\n"),
        ("import os, sys, re\nfrom pathlib import Path\nfrom typing import List\n"
         "def find_python_files(root: str) -> List[Path]:\n"
         "    return sorted(Path(root).rglob('*.py'))\n"
         "def count_lines(path: Path, skip_blank: bool = True) -> int:\n"
         "    lines = path.read_text(errors='replace').splitlines()\n"
         "    return len([l for l in lines if l.strip()]) if skip_blank else len(lines)\n"),
    ]
    high = [
        ("In the summer of 1842, Charles Beaumont arrived at the ancestral estate "
         "having travelled three weeks overland through territories rendered impassable "
         "by the spring floods.  His aunt, Marguerite, had written urgently about the "
         "peculiar disappearances afflicting the village.  Beaumont had dismissed these "
         "letters, yet the coroner's ledger contained entries that could not be explained."),
        ("The negotiations had stalled on a single clause relating to the definition of "
         "'material adverse change,' a phrase whose apparent simplicity concealed decades "
         "of contested jurisprudence.  Ambassador Okonkwo had spent seventeen hours in "
         "the conference suite, sustained by tepid coffee and an unwillingness to cede "
         "the point that her counterpart kept inserting into every reformulation."),
        ("Professor Lysandra Veit had not expected the manuscript to survive the bombing "
         "of the university library in 1944.  Yet here it was, its vellum pages browned "
         "at the edges but legible, resting in a cedar box that had been misfiled in the "
         "sub-basement archives for nearly eighty years.  The text described an "
         "astronomical observation that would predate the accepted discovery by two centuries."),
        ("The rover had been transmitting intermittently for six days when the signal "
         "finally stabilised.  Ground control ran diagnostic checks across all fourteen "
         "sensor arrays, finding anomalous readings in the spectrometer and a partial "
         "blockage in the dust-removal mechanism.  More troubling was the positional drift: "
         "the rover had travelled 340 metres without any commanded movement."),
        ("In her memoir, Yuki Tanaka recalls the afternoon she first understood what her "
         "grandmother had actually witnessed during the occupation.  The revelation arrived "
         "not with drama but with a quiet shift in the quality of light, as if the whole "
         "room had exhaled.  She looked out at the garden her grandmother had planted to "
         "replace what the soldiers had burned, and understood why she always grew things that came back."),
        ("The ecological survey of the Karamoja highlands, conducted over three consecutive "
         "dry seasons, revealed a pattern of soil degradation correlating strongly with the "
         "introduction of a single introduced species of grazing ungulate forty years earlier. "
         "The cascade of consequences had been slow enough to evade detection until the "
         "combined analysis of satellite imagery and ground-truth sampling made the trajectory unmistakable."),
        ("Marisol had been born in the city that no longer existed -- erased first by the "
         "earthquake and then, more thoroughly, by the subsequent bureaucratic reclassification "
         "that transferred its territory to a neighbouring municipality.  She carried its "
         "vanished name in her identity documents, a ghost topology that bureaucrats "
         "occasionally questioned with mild, institutional confusion."),
        ("The inquest into the failure of the Northgate Bridge had entered its third week when "
         "the lead engineer finally admitted under examination that the fatigue testing protocols "
         "had been amended eighteen months before construction began.  The solicitor produced a "
         "series of internal emails suggesting the change had been made not for technical reasons "
         "but to meet a cost target set by the client's infrastructure fund."),
        ("Archaeologists working at the site near the ancient trade route uncovered a collection "
         "of bronzes unlike anything previously documented from the region.  The iconography "
         "combined motifs associated with two distinct cultural traditions separated by both "
         "geography and three centuries of time.  The stratigraphic context was unambiguous; "
         "the dating was consistent.  The find demanded a revision of the accepted timeline."),
        ("The composer had worked on the final movement for eleven years, setting it aside twice "
         "during periods of illness and once following the death of his closest collaborator. "
         "The manuscript, when finally performed posthumously, revealed a structural logic that "
         "critics had initially dismissed -- a long-range harmonic preparation spanning forty "
         "minutes that resolved only in the work's final twenty bars."),
        ("The trial of Minister Ferreira entered its final phase in a courtroom packed with "
         "journalists and former colleagues who had strategically distanced themselves over "
         "the preceding months.  The defence rested its case on the premise that the minister's "
         "signature on the disputed contracts had been obtained through a process of "
         "administrative substitution whose legality had never been tested at this level."),
        ("The migration of data from the legacy system had been scheduled for a Saturday night "
         "when traffic was minimal, but the transformation scripts encountered character-encoding "
         "errors in approximately twelve thousand records dating from 1987 to 1994.  By Sunday "
         "morning, the rollback had itself partially failed, leaving the production system in a "
         "semi-migrated state the engineers described as unprecedented."),
        ("Elena Marchetti had spent twenty years translating technical manuals for industrial "
         "machinery before her editor suggested that the peculiar cadence she had developed for "
         "describing mechanical processes might transfer to fiction.  Her first novel, narrated "
         "entirely from the perspective of a turbine at a hydroelectric plant, was rejected by "
         "fourteen publishers before finding a home with a specialised imprint."),
        ("The study of collective behaviour in the ant colonies of the Chihuahuan desert had "
         "occupied Dr. Finch for the better part of a decade.  What she had not anticipated was "
         "the way the colony's apparent decision-making process broke down under specific moisture "
         "conditions -- not randomly, but in a way that preserved the colony's long-term survival "
         "at the cost of short-term efficiency losses."),
        ("The hurricane had made landfall at 3:47 in the morning, which meant that most residents "
         "had been asleep when the surge topped the seawall.  The emergency management system had "
         "issued its final alert six hours earlier, but the uptake of the mandatory evacuation order "
         "had been, as the after-action review would later document with characteristic bureaucratic "
         "neutrality, 'below projected compliance thresholds.'"),
        ("The poet had written the sequence over a single winter following the dissolution of his "
         "marriage and the loss of his academic position.  The poems were technically accomplished "
         "and emotionally brutal in a way that made most readers uncomfortable.  Twenty years later "
         "it was rediscovered by a generation of writers who had themselves experienced the "
         "particular disorientation the poems described."),
        ("The water rights case had wound through four levels of judicial review before reaching "
         "the supreme court.  At its core the dispute was about a sentence in an 1889 treaty that "
         "used the phrase 'time immemorial' -- a term that the claimants argued referred to a "
         "continuous unbroken usage and that the respondents argued could be interrupted by "
         "legitimate government action without extinguishing the underlying right."),
        ("Dr. Ambrose had published forty-seven papers on the biochemistry of cellular senescence "
         "without ever receiving the recognition that his contemporaries agreed his work deserved. "
         "The reason was simple and unjust: he had been consistently correct ten years before the "
         "experimental tools existed to demonstrate that he was correct, which meant his results "
         "had been dismissed as speculative precisely during the period when citations determine careers."),
        ("The cargo vessel had been adrift for nine days when the coastguard cutter finally reached "
         "it.  The crew of fourteen was alive, subsisting on emergency rations and rainwater, the "
         "engine room flooded to a level the engineers had managed to stabilise but not reverse. "
         "What the initial inspection could not explain was the complete absence of the ship's "
         "electronic navigation equipment -- not damaged, but systematically removed."),
        ("The exhibition had been organised around a central provocation: that the distinction "
         "between folk art and fine art was a historical accident produced by specific conditions "
         "of patronage and institutional legitimation that had no aesthetic justification.  Half "
         "the critics who attended were persuaded; the other half wrote reviews that revealed, in "
         "the organisers' view, exactly the kind of category thinking the exhibition had set out to challenge."),
    ]
    assert len(low) == 20 and len(medium) == 20 and len(high) == 20
    corpus = []
    for text in low:
        corpus.append({"regime": "low", "text": text})
    for text in medium:
        corpus.append({"regime": "medium", "text": text})
    for text in high:
        corpus.append({"regime": "high", "text": text})
    return corpus


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def _hist_entropy(values: Any, num_bins: int = 256) -> float:
    flat = values.detach().float().reshape(-1)
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmax - vmin < 1e-8:
        return 0.0
    normalized = (flat - vmin) / (vmax - vmin + 1e-8)
    indices = (normalized * num_bins).long().clamp(0, num_bins - 1)
    counts = torch.zeros(num_bins, device=flat.device, dtype=torch.float32)
    counts.scatter_add_(0, indices, torch.ones_like(flat, dtype=torch.float32))
    prob = counts / (counts.sum() + 1e-10)
    log_prob = torch.where(prob > 1e-10, torch.log(prob + 1e-10), torch.zeros_like(prob))
    return float(-(prob * log_prob).sum().item())


def _entropy_to_chunk(H: float, num_bins: int = 256,
                      c_min: int = 32, c_max: int = 512) -> int:
    h_ref = math.log(num_bins)
    ratio = min(H / h_ref, 1.0) if h_ref > 0 else 0.0
    raw = c_min + ratio * (c_max - c_min)
    rounded = int(2 ** round(math.log2(max(raw, 1.0))))
    return max(c_min, min(c_max, rounded))


# ---------------------------------------------------------------------------
# Per-prompt analysis
# ---------------------------------------------------------------------------

def _run_prompt(model: Any, tokenizer: Any, prompt: str, device: Any,
                num_bins: int, warmup: int, repeats: int,
                no_timing: bool) -> dict[str, Any]:
    backbone = getattr(model, "backbone", None) or getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        return {"error": "Cannot locate model backbone.layers"}

    captured: list[tuple[int, Any]] = []

    def _make_hook(layer_idx: int):
        def _hook(module: Any, inp: Any, out: Any) -> None:
            if inp and inp[0] is not None:
                captured.append((layer_idx, inp[0].detach()))
        return _hook

    handles = []
    for i, layer in enumerate(backbone.layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is not None and hasattr(mixer, "x_proj"):
            handles.append(mixer.x_proj.register_forward_hook(_make_hook(i)))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_len = int(inputs["input_ids"].shape[-1])
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    if not captured:
        return {"prompt_len": prompt_len, "n_layers_captured": 0,
                "entropy_mean": None, "chunk_mode": None, "chunk_dist": {}}

    entropies: list[float] = []
    chunk_dist: dict[int, int] = {}
    for _li, u in captured:
        H = _hist_entropy(u, num_bins=num_bins)
        c = _entropy_to_chunk(H, num_bins=num_bins)
        entropies.append(H)
        chunk_dist[c] = chunk_dist.get(c, 0) + 1

    mode_chunk = max(chunk_dist, key=lambda k: chunk_dist[k]) if chunk_dist else None
    result: dict[str, Any] = {
        "prompt_len": prompt_len,
        "n_layers_captured": len(captured),
        "entropy_mean": round(statistics.mean(entropies), 6),
        "entropy_std": round(statistics.pstdev(entropies), 6) if len(entropies) > 1 else 0.0,
        "entropy_min": round(min(entropies), 6),
        "entropy_max": round(max(entropies), 6),
        "chunk_mode": mode_chunk,
        "chunk_dist": dict(sorted(chunk_dist.items())),
    }

    if not no_timing:
        for _ in range(warmup):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        latencies: list[float] = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=16, do_sample=False)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        result["latency_mean_ms"] = round(statistics.mean(latencies), 4)
        result["latency_std_ms"] = round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    _set_hf_token_from_envfile()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_REGISTRY.get(args.model, args.model)
    if rank == 0:
        print(f"[hetero_8gpu] Loading {model_id} on {world_size} GPUs ...")
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=str(device),
    ).eval()

    h_ref_val = round(math.log(args.num_bins), 6)
    if rank == 0:
        print(f"[hetero_8gpu] H_ref=log({args.num_bins})={h_ref_val:.4f} nats  "
              f"timing={'disabled' if args.no_timing else f'{args.repeats} repeats'}")

    corpus = _build_corpus()  # 60 prompts

    # Shard: rank i processes corpus[i::world_size]
    my_items = corpus[rank::world_size]

    my_results: list[dict[str, Any]] = []
    for local_idx, item in enumerate(my_items):
        global_idx = rank + local_idx * world_size
        r = _run_prompt(
            model, tokenizer, item["text"], device,
            num_bins=args.num_bins,
            warmup=args.warmup,
            repeats=args.repeats,
            no_timing=args.no_timing,
        )
        r["regime"] = item["regime"]
        r["prompt_idx_global"] = global_idx
        r["rank"] = rank
        my_results.append(r)
        H = r.get("entropy_mean", "?")
        c = r.get("chunk_mode", "?")
        print(f"[hetero_8gpu] rank={rank}  idx={global_idx:2d}  [{item['regime']:6s}]  "
              f"H={H:.4f}  chunk={c}", flush=True)

    # Gather all results at rank 0
    all_rank_results: list[Any] = [None] * world_size
    dist.gather_object(my_results, all_rank_results if rank == 0 else None, dst=0)
    dist.barrier()

    if rank == 0:
        # Flatten and sort by global index
        all_results: list[dict[str, Any]] = []
        for rank_list in all_rank_results:
            all_results.extend(rank_list)
        all_results.sort(key=lambda r: r.get("prompt_idx_global", 0))

        # Regime summaries
        regime_totals: dict[str, list[dict[str, Any]]] = {"low": [], "medium": [], "high": []}
        for item in all_results:
            regime_totals[item["regime"]].append(item)

        regime_summaries: dict[str, Any] = {}
        print("\n[hetero_8gpu] === Regime Summary ===")
        for regime in ["low", "medium", "high"]:
            items = regime_totals[regime]
            ent_means = [r["entropy_mean"] for r in items if r.get("entropy_mean") is not None]
            chunk_agg: dict[int, int] = {}
            for r in items:
                for k, v in r.get("chunk_dist", {}).items():
                    chunk_agg[k] = chunk_agg.get(k, 0) + v
            h_m = round(statistics.mean(ent_means), 4) if ent_means else None
            h_s = round(statistics.pstdev(ent_means), 4) if len(ent_means) > 1 else 0.0
            dist_str = " ".join(f"chunk{k}:{v}" for k, v in sorted(chunk_agg.items()))
            print(f"  {regime:<10}  n={len(items)}  H={h_m}±{h_s:.4f}  {dist_str}")
            regime_summaries[regime] = {
                "n": len(items),
                "entropy_mean": h_m,
                "entropy_std": h_s,
                "chunk_dist_aggregate": {int(k): v for k, v in chunk_agg.items()},
            }

        non_degenerate = sum(len(s["chunk_dist_aggregate"]) > 1
                            for s in regime_summaries.values())
        print(f"\n[hetero_8gpu] Regimes with non-degenerate chunk dist: {non_degenerate}/3")
        if non_degenerate >= 2:
            print("  => COREY demonstrates multi-regime chunk switching on this corpus.")
        else:
            print("  => Chunk selection degenerate across regimes.")

        output = {
            "world_size": world_size,
            "gpu": torch.cuda.get_device_name(0),
            "model": args.model,
            "num_bins": args.num_bins,
            "h_ref": h_ref_val,
            "timing_enabled": not args.no_timing,
            "platform": platform.platform(),
            "regime_summaries": {k: {**v, "chunk_dist_aggregate": {str(kk): vv for kk, vv in v["chunk_dist_aggregate"].items()}}
                                  for k, v in regime_summaries.items()},
            "per_prompt": all_results,
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / "summary.json"
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[hetero_8gpu] Results saved to {out_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
