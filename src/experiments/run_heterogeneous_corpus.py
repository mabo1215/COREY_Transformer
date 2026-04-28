"""
Heterogeneous Corpus Evaluation: Cross-Regime Chunk Switching.

Fills the TODO rows in Table ``tab:heterogeneous`` (main text §7.5):
  - Low-entropy (templated / repetitive):  20 prompts
  - Medium-entropy (code):                 20 prompts
  - High-entropy (dense natural language): 20 prompts

For each prompt, runs Mamba-370M with the active entropy hook (inline entropy
computation, chunk selection logged per layer) and records:
  - mean entropy across all instrumented layers
  - chunk selected by COREY (H_ref = log K, default calibration)
  - latency per prompt

The per-regime chunk distribution validates COREY as an adaptive controller
(non-degenerate distribution across buckets) rather than a single-regime
auto-tuner.

Works with or without mamba_ssm CUDA kernels.  When kernels are unavailable
the HF Python sequential scan is used; latency figures reflect that fallback
rather than true hardware performance, but entropy and chunk selection are
accurate regardless.

Usage:
    # With mamba_ssm CUDA kernels (adama-cuda128):
    python -m src.experiments.run_heterogeneous_corpus \\
        --model mamba-370m --warmup 1 --repeats 3 \\
        --output-dir src/outputs/heterogeneous_corpus

    # Without mamba_ssm (entropy-only mode, no reliable latency):
    python -m src.experiments.run_heterogeneous_corpus \\
        --model mamba-370m --no-timing \\
        --output-dir src/outputs/heterogeneous_corpus
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
try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


MODEL_REGISTRY: dict[str, str] = {
    "mamba-370m": "state-spaces/mamba-370m-hf",
    "mamba-1.4b": "state-spaces/mamba-1.4b-hf",
}


# ---------------------------------------------------------------------------
# 60-prompt heterogeneous corpus
# ---------------------------------------------------------------------------

def _build_corpus() -> list[dict[str, str]]:
    """Return 60 prompts: 20 low / 20 medium / 20 high entropy regime."""

    # --- Low-entropy: templated / repetitive ---
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

    # --- Medium-entropy: Python code docstrings ---
    medium = [
        (
            "def binary_search(arr, target):\n"
            "    \"\"\"Search sorted array for target using binary search.\n"
            "    Returns index if found, -1 otherwise. Time: O(log n).\"\"\"\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
        ),
        (
            "class LinkedList:\n"
            "    \"\"\"Singly linked list with O(1) prepend and O(n) search.\"\"\"\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
            "        self.size = 0\n"
            "    def prepend(self, val):\n"
            "        node = Node(val, self.head)\n"
            "        self.head = node\n"
            "        self.size += 1\n"
        ),
        (
            "import torch\nimport torch.nn as nn\n\n"
            "class LayerNorm(nn.Module):\n"
            "    \"\"\"Layer normalization over last dimension.\"\"\"\n"
            "    def __init__(self, d_model, eps=1e-6):\n"
            "        super().__init__()\n"
            "        self.gamma = nn.Parameter(torch.ones(d_model))\n"
            "        self.beta = nn.Parameter(torch.zeros(d_model))\n"
            "        self.eps = eps\n"
        ),
        (
            "def merge_sort(arr):\n"
            "    \"\"\"Stable O(n log n) sort via divide and conquer.\"\"\"\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    mid = len(arr) // 2\n"
            "    left = merge_sort(arr[:mid])\n"
            "    right = merge_sort(arr[mid:])\n"
            "    return merge(left, right)\n"
        ),
        (
            "class Tokenizer:\n"
            "    \"\"\"BPE tokenizer with vocabulary and merge rules.\"\"\"\n"
            "    def __init__(self, vocab_size=50000):\n"
            "        self.vocab = {}\n"
            "        self.merges = []\n"
            "        self.vocab_size = vocab_size\n"
            "    def encode(self, text):\n"
            "        tokens = list(text.encode('utf-8'))\n"
            "        return self._apply_merges(tokens)\n"
        ),
        (
            "def attention(Q, K, V, mask=None):\n"
            "    \"\"\"Scaled dot-product attention. Q,K,V: (B, H, L, d).\"\"\"\n"
            "    import math\n"
            "    d_k = Q.size(-1)\n"
            "    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)\n"
            "    if mask is not None:\n"
            "        scores = scores.masked_fill(mask == 0, -1e9)\n"
            "    return torch.softmax(scores, dim=-1) @ V\n"
        ),
        (
            "class DataLoader:\n"
            "    \"\"\"Batched iterator over a dataset with shuffling.\"\"\"\n"
            "    def __init__(self, dataset, batch_size=32, shuffle=True):\n"
            "        self.dataset = dataset\n"
            "        self.batch_size = batch_size\n"
            "        self.shuffle = shuffle\n"
            "        self.indices = list(range(len(dataset)))\n"
            "    def __iter__(self):\n"
            "        if self.shuffle:\n"
            "            import random\n"
            "            random.shuffle(self.indices)\n"
        ),
        (
            "def compute_iou(box1, box2):\n"
            "    \"\"\"Compute intersection-over-union for two axis-aligned boxes.\"\"\"\n"
            "    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])\n"
            "    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])\n"
            "    inter = max(0, x2 - x1) * max(0, y2 - y1)\n"
            "    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])\n"
            "    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])\n"
            "    return inter / (area1 + area2 - inter + 1e-6)\n"
        ),
        (
            "import numpy as np\n\n"
            "def pca(X, n_components=2):\n"
            "    \"\"\"PCA via SVD. X: (n_samples, n_features).\"\"\"\n"
            "    X_centered = X - X.mean(axis=0)\n"
            "    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)\n"
            "    return X_centered @ Vt[:n_components].T\n"
        ),
        (
            "class RingBuffer:\n"
            "    \"\"\"Fixed-capacity circular buffer with O(1) append.\"\"\"\n"
            "    def __init__(self, capacity):\n"
            "        self.buf = [None] * capacity\n"
            "        self.capacity = capacity\n"
            "        self.head = 0\n"
            "        self.count = 0\n"
            "    def append(self, val):\n"
            "        self.buf[self.head % self.capacity] = val\n"
            "        self.head += 1\n"
            "        self.count = min(self.count + 1, self.capacity)\n"
        ),
        (
            "def dijkstra(graph, source):\n"
            "    \"\"\"Single-source shortest paths via min-heap.\"\"\"\n"
            "    import heapq\n"
            "    dist = {v: float('inf') for v in graph}\n"
            "    dist[source] = 0\n"
            "    heap = [(0, source)]\n"
            "    while heap:\n"
            "        d, u = heapq.heappop(heap)\n"
            "        if d > dist[u]:\n"
            "            continue\n"
            "        for v, w in graph[u]:\n"
            "            if dist[u] + w < dist[v]:\n"
            "                dist[v] = dist[u] + w\n"
            "                heapq.heappush(heap, (dist[v], v))\n"
            "    return dist\n"
        ),
        (
            "class GradientDescent:\n"
            "    \"\"\"Vanilla gradient descent optimizer.\"\"\"\n"
            "    def __init__(self, lr=0.01):\n"
            "        self.lr = lr\n"
            "    def step(self, params, grads):\n"
            "        return [p - self.lr * g for p, g in zip(params, grads)]\n"
            "    def zero_grad(self):\n"
            "        pass\n"
        ),
        (
            "def tokenize_sentence(sentence):\n"
            "    \"\"\"Simple whitespace + punctuation tokenizer.\"\"\"\n"
            "    import re\n"
            "    tokens = re.findall(r\"\\b\\w+\\b|[.,!?;:]\", sentence.lower())\n"
            "    return tokens\n\n"
            "def build_vocab(corpus):\n"
            "    vocab = {}\n"
            "    for sentence in corpus:\n"
            "        for token in tokenize_sentence(sentence):\n"
            "            vocab[token] = vocab.get(token, 0) + 1\n"
            "    return vocab\n"
        ),
        (
            "import hashlib\n\n"
            "class BloomFilter:\n"
            "    \"\"\"Probabilistic set membership test.\"\"\"\n"
            "    def __init__(self, capacity, error_rate=0.01):\n"
            "        m = -capacity * math.log(error_rate) / math.log(2)**2\n"
            "        self.size = int(m)\n"
            "        self.bits = bytearray(self.size)\n"
            "        self.k = int(self.size / capacity * math.log(2))\n"
            "    def add(self, item):\n"
            "        for i in range(self.k):\n"
            "            h = int(hashlib.md5(f\"{item}{i}\".encode()).hexdigest(), 16)\n"
            "            self.bits[h % self.size] = 1\n"
        ),
        (
            "def cross_entropy(logits, targets):\n"
            "    \"\"\"Numerically stable cross-entropy loss.\"\"\"\n"
            "    import torch.nn.functional as F\n"
            "    log_probs = F.log_softmax(logits, dim=-1)\n"
            "    return -log_probs[range(len(targets)), targets].mean()\n\n"
            "def accuracy(logits, targets):\n"
            "    preds = logits.argmax(dim=-1)\n"
            "    return (preds == targets).float().mean().item()\n"
        ),
        (
            "class ConvBlock(nn.Module):\n"
            "    \"\"\"Conv2d + BN + ReLU building block.\"\"\"\n"
            "    def __init__(self, in_ch, out_ch, kernel=3, stride=1):\n"
            "        super().__init__()\n"
            "        pad = kernel // 2\n"
            "        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)\n"
            "        self.bn = nn.BatchNorm2d(out_ch)\n"
            "    def forward(self, x):\n"
            "        return torch.relu(self.bn(self.conv(x)))\n"
        ),
        (
            "def softmax_temperature(logits, T=1.0):\n"
            "    \"\"\"Temperature-scaled softmax for sampling diversity control.\"\"\"\n"
            "    import torch\n"
            "    scaled = logits / T\n"
            "    return torch.softmax(scaled, dim=-1)\n\n"
            "def top_k_filter(logits, k=50):\n"
            "    v, _ = torch.topk(logits, k)\n"
            "    return logits.masked_fill(logits < v[..., [-1]], -float('inf'))\n"
        ),
        (
            "class LRUCache:\n"
            "    \"\"\"O(1) get/put cache with least-recently-used eviction.\"\"\"\n"
            "    def __init__(self, capacity):\n"
            "        from collections import OrderedDict\n"
            "        self.cache = OrderedDict()\n"
            "        self.capacity = capacity\n"
            "    def get(self, key):\n"
            "        if key not in self.cache:\n"
            "            return -1\n"
            "        self.cache.move_to_end(key)\n"
            "        return self.cache[key]\n"
        ),
        (
            "def beam_search(model, input_ids, beam_width=5, max_len=50):\n"
            "    \"\"\"Beam search decoding for sequence generation.\"\"\"\n"
            "    import torch\n"
            "    beams = [(0.0, input_ids.tolist())]\n"
            "    for _ in range(max_len):\n"
            "        candidates = []\n"
            "        for score, seq in beams:\n"
            "            ids = torch.tensor([seq])\n"
            "            with torch.no_grad():\n"
            "                logits = model(ids).logits[0, -1]\n"
            "            probs = torch.log_softmax(logits, dim=-1)\n"
            "            top = probs.topk(beam_width)\n"
            "            for p, idx in zip(top.values, top.indices):\n"
            "                candidates.append((score + p.item(), seq + [idx.item()]))\n"
            "        beams = sorted(candidates, reverse=True)[:beam_width]\n"
        ),
        (
            "import os, sys, re\n"
            "from pathlib import Path\n"
            "from typing import Optional, List, Dict, Tuple\n\n"
            "def find_python_files(root: str, pattern: str = '*.py') -> List[Path]:\n"
            "    \"\"\"Recursively find Python source files matching pattern.\"\"\"\n"
            "    return sorted(Path(root).rglob(pattern))\n\n"
            "def count_lines(path: Path, skip_blank: bool = True) -> int:\n"
            "    lines = path.read_text(errors='replace').splitlines()\n"
            "    if skip_blank:\n"
            "        lines = [l for l in lines if l.strip()]\n"
            "    return len(lines)\n"
        ),
    ]

    # --- High-entropy: dense narrative / NarrativeQA-style passages ---
    high = [
        (
            "In the summer of 1842, Charles Beaumont arrived at the ancestral estate "
            "having travelled three weeks overland through territories rendered "
            "impassable by the spring floods.  His aunt, Marguerite, had written "
            "urgently about the peculiar disappearances afflicting the village — "
            "seven men gone without a trace, the last seen walking toward the marshes "
            "on a moonless Thursday.  Beaumont had dismissed these letters as the "
            "confabulations of an aging mind, yet the coroner's ledger he now held "
            "contained entries that could not be easily explained away."
        ),
        (
            "The negotiations had stalled on a single clause relating to the "
            "definition of 'material adverse change,' a phrase whose apparent "
            "simplicity concealed decades of contested jurisprudence.  Ambassador "
            "Okonkwo had spent seventeen hours in the conference suite, sustained "
            "by tepid coffee and an unwillingness to cede the point that her "
            "counterpart, a veteran diplomat from the opposing bloc, kept inserting "
            "into every reformulation.  Outside, protesters had gathered again, "
            "their chants audible even through the triple-glazed windows."
        ),
        (
            "Professor Lysandra Veit had not expected the manuscript to survive "
            "the bombing of the university library in 1944.  Yet here it was, its "
            "vellum pages browned at the edges but legible, the ink faded but not "
            "lost, resting in a cedar box that had been misfiled in the sub-basement "
            "archives for nearly eighty years.  The text, in a dialect of Middle "
            "Low German with traces of Latin marginalia, described an astronomical "
            "observation that, if accurate, would predate the commonly accepted "
            "discovery of the phenomenon by more than two centuries."
        ),
        (
            "The rover had been transmitting intermittently for six days when the "
            "signal finally stabilised.  Ground control at the Pasadena facility "
            "ran diagnostic checks across all fourteen sensor arrays, finding "
            "anomalous readings in the spectrometer and a partial blockage in the "
            "dust-removal mechanism.  More troubling was the positional drift: "
            "the rover had travelled 340 metres from its last confirmed location "
            "without any commanded movement.  The engineers argued through the night, "
            "their hypotheses ranging from software fault to an improbable seismic "
            "event on the plateau."
        ),
        (
            "In her memoir, Yuki Tanaka recalls the afternoon she first understood "
            "what her grandmother had actually witnessed during the occupation — not "
            "the sanitised account repeated at family dinners, but the unedited truth "
            "preserved in a journal nobody had thought to translate.  The revelation "
            "arrived not with drama but with a quiet shift in the quality of light, "
            "as if the whole room had exhaled.  She set down the translated pages "
            "and looked out at the garden her grandmother had planted to replace "
            "what the soldiers had burned, and understood at last why she had always "
            "chosen to grow things that came back."
        ),
        (
            "The ecological survey of the Karamoja highlands, conducted over three "
            "consecutive dry seasons, revealed a pattern of soil degradation "
            "correlating strongly with the introduction of a single introduced "
            "species of grazing ungulate forty years earlier.  The cascade of "
            "consequences had been slow enough to evade detection until the combined "
            "analysis of satellite imagery and ground-truth sampling made the "
            "trajectory unmistakable.  Reversing the damage, the report concluded, "
            "would require not only removal of the introduced species but active "
            "restoration of the nitrogen-fixing shrubs that had once anchored "
            "the plateau's thin topsoil."
        ),
        (
            "Marisol had been born in the city that no longer existed — erased first "
            "by the earthquake and then, more thoroughly, by the subsequent "
            "bureaucratic reclassification that transferred its territory to a "
            "neighbouring municipality.  She carried its vanished name in her "
            "identity documents, a ghost topology that bureaucrats occasionally "
            "questioned, and she had grown skilled at explaining the particular "
            "ontological status of a place that was simultaneously real in memory "
            "and absent from every official map."
        ),
        (
            "The inquest into the failure of the Northgate Bridge had entered its "
            "third week when the lead engineer finally admitted under examination "
            "that the fatigue testing protocols had been amended eighteen months "
            "before construction began.  The amendment, authorised by a deputy "
            "director who had since resigned, reduced the required load cycles by "
            "thirty percent.  The solicitor representing the families of the eleven "
            "people who had died in the collapse produced a series of internal "
            "emails that suggested the change had been made not for technical reasons "
            "but to meet a cost target set by the client's infrastructure fund."
        ),
        (
            "Archaeologists working at the site near the ancient trade route "
            "uncovered a collection of bronzes unlike anything previously documented "
            "from the region.  The iconography combined motifs associated with "
            "two distinct cultural traditions separated, according to prevailing "
            "theory, by both geography and three centuries of time.  The stratigraphic "
            "context was unambiguous; the dating was consistent.  The find either "
            "demanded a revision of the timeline of cultural contact in the region "
            "or suggested the existence of a previously unknown intermediary "
            "tradition that had borrowed freely from both."
        ),
        (
            "The composer had worked on the final movement for eleven years, setting "
            "it aside twice during periods of illness and once following the death "
            "of his closest collaborator.  The manuscript, when it was finally "
            "performed posthumously, revealed a structural logic that critics had "
            "initially dismissed as incoherence — a long-range harmonic preparation "
            "spanning forty minutes of music that resolved only in the work's final "
            "twenty bars.  Several listeners reported that they had wept without "
            "entirely understanding why."
        ),
        (
            "The trial of Minister Ferreira entered its final phase in a courtroom "
            "packed with journalists, activists, and former colleagues who had "
            "strategically distanced themselves over the preceding months.  The "
            "defence rested its case on the premise that the minister's signature "
            "on the disputed contracts had been obtained through a process of "
            "administrative substitution — a procedural mechanism whose legality "
            "had never been tested at this level of government.  The prosecution "
            "countered that legality and culpability were not synonymous terms, "
            "and that the evidence of personal enrichment was independent of the "
            "chain-of-signature question."
        ),
        (
            "The migration of data from the legacy system had been scheduled for "
            "a Saturday night in November when traffic was minimal, but the "
            "transformation scripts encountered character-encoding errors in "
            "approximately twelve thousand records dating from 1987 to 1994 — "
            "the years when the agency had operated two incompatible database "
            "platforms simultaneously.  By Sunday morning, the rollback had "
            "itself partially failed, leaving the production system in a "
            "semi-migrated state that the on-call engineers described, with "
            "understandable frustration, as unprecedented."
        ),
        (
            "Elena Marchetti had spent twenty years translating technical manuals "
            "for industrial machinery before her editor at the small press suggested "
            "that the peculiar cadence she had developed for describing mechanical "
            "processes might transfer to fiction.  Her first novel, narrated "
            "entirely from the perspective of a turbine at a hydroelectric plant, "
            "was rejected by fourteen publishers before finding a home with an "
            "imprint that specialised in what its catalogue described as 'fiction "
            "that requires patience from its reader.'"
        ),
        (
            "The study of collective behaviour in the ant colonies of the "
            "Chihuahuan desert had occupied Dr. Finch for the better part of "
            "a decade.  She had documented the foraging patterns, the division "
            "of labour, the chemical signalling cascades.  What she had not "
            "anticipated was the way the colony's apparent decision-making "
            "process broke down under specific moisture conditions — not randomly, "
            "but in a way that preserved the colony's long-term survival at the "
            "cost of short-term efficiency losses that would have bankrupted any "
            "human enterprise operating under comparable constraints."
        ),
        (
            "The hurricane had made landfall at 3:47 in the morning, which meant "
            "that most residents of the coastal township had been asleep when the "
            "surge topped the seawall.  The emergency management system had issued "
            "its final alert six hours earlier, but the uptake of the mandatory "
            "evacuation order had been, as the after-action review would later "
            "document with characteristic bureaucratic neutrality, 'below projected "
            "compliance thresholds.'  Forty-three people had chosen to shelter in "
            "place.  Forty-one of them survived."
        ),
        (
            "The poet had written the sequence over a single winter following "
            "the dissolution of his marriage and the loss of his academic position.  "
            "The poems were technically accomplished and emotionally brutal in a "
            "way that made most readers uncomfortable.  The collection won a prize, "
            "sold poorly, and was out of print within three years.  Twenty years "
            "later it was rediscovered by a generation of writers who had themselves "
            "experienced the particular disorientation the poems described, and it "
            "entered the curriculum of several universities as an example of a "
            "work that had arrived too early for the audience it needed."
        ),
        (
            "The water rights case had wound through four levels of judicial "
            "review before reaching the supreme court, where it had been pending "
            "for eleven months.  At its core the dispute was about a sentence "
            "in an 1889 treaty that used the phrase 'time immemorial' — a term "
            "that the claimants argued referred to a continuous unbroken usage "
            "and that the respondents argued referred to nothing more than a "
            "customary practice that could be interrupted by legitimate government "
            "action without extinguishing the underlying right."
        ),
        (
            "Dr. Ambrose had published forty-seven papers on the biochemistry of "
            "cellular senescence without ever receiving the recognition that his "
            "contemporaries agreed his work deserved.  The reason was simple and "
            "unjust: he had been consistently correct ten years before the "
            "experimental tools existed to demonstrate that he was correct, "
            "which meant that his results had been dismissed as speculative "
            "precisely during the period when citation practices determine "
            "career trajectories."
        ),
        (
            "The cargo vessel had been adrift for nine days when the coastguard "
            "cutter finally reached it.  The crew of fourteen was alive, "
            "subsisting on emergency rations and rainwater, the engine room "
            "flooded to a level that the engineers had managed to stabilise "
            "but not reverse.  What the initial inspection could not explain "
            "was the complete absence of the ship's electronic navigation "
            "equipment — not damaged or destroyed, but systematically removed, "
            "along with the logbook and all copies of the cargo manifest."
        ),
        (
            "The exhibition had been organised around a central provocation: "
            "that the distinction between folk art and fine art was a historical "
            "accident produced by specific conditions of patronage and institutional "
            "legitimation that had no aesthetic justification.  Half the critics "
            "who attended the opening were persuaded; the other half wrote reviews "
            "that revealed, in the organisers' view, exactly the kind of category "
            "thinking the exhibition had set out to challenge.  Both groups, "
            "remarkably, gave the show positive coverage."
        ),
        (
            "The forensic accountants had been working through the subsidiary "
            "structure for six weeks before they located the mechanism: a series "
            "of intracompany loans issued at below-market interest rates that, "
            "when traced through four jurisdictions, resolved into a pattern of "
            "value extraction that had been invisible in any single jurisdiction's "
            "accounts but was unmistakable when the consolidated picture emerged.  "
            "The total sum was modest by the standards of corporate fraud, but "
            "the elegance of the construction attracted a certain grim admiration "
            "from the regulators who had spent the better part of a year failing "
            "to find it."
        ),
    ]

    assert len(low) >= 20, f"Expected at least 20 low-entropy prompts, got {len(low)}"
    assert len(medium) >= 20, f"Expected at least 20 medium-entropy prompts, got {len(medium)}"
    assert len(high) >= 20, f"Expected at least 20 high-entropy prompts, got {len(high)}"
    low = low[:20]
    medium = medium[:20]
    high = high[:20]

    corpus = []
    for text in low:
        corpus.append({"regime": "low", "text": text})
    for text in medium:
        corpus.append({"regime": "medium", "text": text})
    for text in high:
        corpus.append({"regime": "high", "text": text})
    return corpus


# ---------------------------------------------------------------------------
# Entropy + COREY chunk selection (H_ref = log K)
# ---------------------------------------------------------------------------

def _hist_entropy(values: Any, num_bins: int = 256) -> float:
    import torch
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

def _run_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: Any,
    *,
    num_bins: int,
    warmup: int,
    repeats: int,
    no_timing: bool,
) -> dict[str, Any]:
    import torch

    backbone = getattr(model, "backbone", None) or getattr(model, "model", None)
    if backbone is None or not hasattr(backbone, "layers"):
        raise RuntimeError("Cannot locate model backbone.layers")

    # Capture post-conv hidden states via forward hooks on all MambaMixer layers.
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
        return {
            "prompt_len": prompt_len,
            "n_layers_captured": 0,
            "entropy_mean": None,
            "entropy_std": None,
            "chunk_mode": None,
            "chunk_dist": {},
        }

    entropies = []
    chunk_dist: dict[int, int] = {}
    for _layer_idx, u in captured:
        # x_proj input is [batch, seq, hidden] — use all
        H = _hist_entropy(u, num_bins=num_bins)
        c = _entropy_to_chunk(H, num_bins=num_bins)
        entropies.append(H)
        chunk_dist[c] = chunk_dist.get(c, 0) + 1

    mode_chunk = max(chunk_dist, key=lambda k: chunk_dist[k]) if chunk_dist else None

    result: dict[str, Any] = {
        "prompt_len": prompt_len,
        "n_layers_captured": len(captured),
        "entropy_mean": round(statistics.mean(entropies), 6),
        "entropy_std":  round(statistics.pstdev(entropies), 6) if len(entropies) > 1 else 0.0,
        "entropy_min":  round(min(entropies), 6),
        "entropy_max":  round(max(entropies), 6),
        "chunk_mode":   mode_chunk,
        "chunk_dist":   dict(sorted(chunk_dist.items())),
    }

    if not no_timing:
        for _ in range(warmup):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=16, do_sample=False)
        # Sync after warmup
        if xm:
            xm.mark_step()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies: list[float] = []
        for _ in range(repeats):
            if xm:
                xm.mark_step()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=16, do_sample=False)
            if xm:
                xm.mark_step()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        result["latency_mean_ms"] = round(statistics.mean(latencies), 4)
        result["latency_std_ms"]  = round(statistics.pstdev(latencies), 4) if len(latencies) > 1 else 0.0

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",      default="mamba-370m", choices=list(MODEL_REGISTRY))
    p.add_argument("--num-bins",   type=int, default=256)
    p.add_argument("--warmup",     type=int, default=1)
    p.add_argument("--repeats",    type=int, default=3)
    p.add_argument("--no-timing",  action="store_true",
                   help="Skip latency measurement; only collect entropy/chunk stats.")
    p.add_argument("--output-dir", type=Path, default=Path("src/outputs/heterogeneous_corpus"))
    return p.parse_args()



def _set_hf_token_from_envfile(env_path="/home/amabo1215/source/.env"):
    import os
    try:
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("Huggingface_model_token:="):
                        token = line.strip().split("Huggingface_model_token:=", 1)[-1].strip()
                        if token:
                            os.environ["HF_TOKEN"] = token
                            print(f"[hetero] Set HF_TOKEN from {env_path}")
                        break
    except Exception as e:
        print(f"[hetero] Failed to set HF_TOKEN from {env_path}: {e}")

def _prefer_installed_mamba_kernels() -> None:
    """Make HF Mamba use locally installed CUDA kernels when available."""
    try:
        from types import SimpleNamespace

        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        from transformers.models.mamba import modeling_mamba as _mm
    except Exception as exc:
        print(f"[hetero] Installed mamba kernels unavailable: {exc}")
        return

    original_lazy_load = getattr(_mm, "lazy_load_kernel", None)
    original_resolve = getattr(_mm, "resolve_internal_import", None)
    mamba_kernel = SimpleNamespace(
        selective_scan_fn=selective_scan_fn,
        selective_state_update=selective_state_update,
        mamba_inner_fn=mamba_inner_fn,
    )
    conv_kernel = SimpleNamespace(
        causal_conv1d_fn=causal_conv1d_fn,
        causal_conv1d_update=causal_conv1d_update,
    )

    def lazy_load_kernel(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mamba-ssm":
            return mamba_kernel
        if name == "causal-conv1d":
            return conv_kernel
        if original_lazy_load is not None:
            return original_lazy_load(name, *args, **kwargs)
        return None

    def resolve_internal_import(module: Any, chained_path: str, *args: Any, **kwargs: Any) -> Any:
        if module is mamba_kernel and chained_path.endswith("selective_state_update"):
            return selective_state_update
        if original_resolve is not None:
            return original_resolve(module, chained_path, *args, **kwargs)
        return None

    _mm.lazy_load_kernel = lazy_load_kernel
    _mm.resolve_internal_import = resolve_internal_import
    print("[hetero] Using installed mamba_ssm / causal_conv1d CUDA kernels.")


def main() -> None:
    args = _parse_args()
    _set_hf_token_from_envfile()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    if device.type != "cuda" and not args.no_timing:
        print("[hetero] WARNING: CUDA not available. Latency figures will be unreliable.")

    model_id = MODEL_REGISTRY.get(args.model, args.model)
    print(f"[hetero] Loading {model_id} …")
    _prefer_installed_mamba_kernels()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=str(device),
    ).eval()

    h_ref_val = round(math.log(args.num_bins), 6)
    print(f"[hetero] Device: {gpu_name}  H_ref=log({args.num_bins})={h_ref_val:.4f} nats")
    print(f"[hetero] Timing: {'disabled (--no-timing)' if args.no_timing else f'{args.repeats} repeats'}")
    print()

    corpus = _build_corpus()
    per_prompt_results: list[dict[str, Any]] = []
    regime_totals: dict[str, list[dict[str, Any]]] = {"low": [], "medium": [], "high": []}

    for idx, item in enumerate(corpus):
        regime = item["regime"]
        print(f"[hetero] {idx+1:3d}/60  [{regime:6s}]  ", end="", flush=True)
        r = _run_prompt(
            model, tokenizer, item["text"], device,
            num_bins=args.num_bins,
            warmup=args.warmup,
            repeats=args.repeats,
            no_timing=args.no_timing,
        )
        r["regime"] = regime
        r["prompt_idx"] = idx
        per_prompt_results.append(r)
        regime_totals[regime].append(r)
        H = r.get("entropy_mean", "?")
        c = r.get("chunk_mode", "?")
        lat_str = ""
        if "latency_mean_ms" in r:
            lat_str = f"  lat={r['latency_mean_ms']:.1f}ms"
        print(f"H={H:.4f}  chunk={c}{lat_str}")

    # --- Aggregate per regime ---
    print()
    print("[hetero] === Regime Summary ===")
    print(f"{'Regime':<10} {'N':>4} {'H mean':>8} {'H std':>7} {'chunk dist'}")
    print("-" * 70)

    regime_summaries: dict[str, Any] = {}
    for regime in ["low", "medium", "high"]:
        items = regime_totals[regime]
        ent_means = [r["entropy_mean"] for r in items if r.get("entropy_mean") is not None]
        chunk_mode_all: dict[int, int] = {}
        for r in items:
            for k, v in r.get("chunk_dist", {}).items():
                chunk_mode_all[k] = chunk_mode_all.get(k, 0) + v
        h_m = round(statistics.mean(ent_means), 4) if ent_means else None
        h_s = round(statistics.pstdev(ent_means), 4) if len(ent_means) > 1 else 0.0
        dist_str = " ".join(f"chunk{k}:{v}" for k, v in sorted(chunk_mode_all.items()))
        print(f"{regime:<10} {len(items):>4} {h_m or '?':>8} {h_s:>7.4f}  {dist_str}")
        regime_summaries[regime] = {
            "n": len(items),
            "entropy_mean": h_m,
            "entropy_std": h_s,
            "chunk_dist_aggregate": chunk_mode_all,
        }

    non_degenerate = sum(len(s["chunk_dist_aggregate"]) > 1
                        for s in regime_summaries.values())
    print()
    print(f"[hetero] Regimes with non-degenerate chunk distribution: {non_degenerate}/3")
    if non_degenerate >= 2:
        print("  => COREY demonstrates multi-regime chunk switching on this corpus.")
    else:
        print("  => Chunk selection degenerate: all regimes map to same bucket.")
        print("     This may indicate entropy concentration; review per-prompt values.")

    # --- Paper-ready table ---
    print()
    print("[hetero] === Paper Table (tab:heterogeneous) ===")
    print(f"{'Regime':<30} {'N':>3} {'H mean':>8} {'chunk_mode':>12} {'chunk_dist'}")
    for regime, label in [
        ("low",    "Low-entropy (templated)"),
        ("medium", "Medium-entropy (code)"),
        ("high",   "High-entropy (narrative)"),
    ]:
        s = regime_summaries[regime]
        items = regime_totals[regime]
        chunks = [r["chunk_mode"] for r in items if r.get("chunk_mode") is not None]
        mode = max(set(chunks), key=chunks.count) if chunks else None
        dist_str = " ".join(f"{k}:{v}" for k, v in sorted(s["chunk_dist_aggregate"].items()))
        print(f"{label:<30} {s['n']:>3} {s['entropy_mean'] or '?':>8}  {mode or '?':>12}  {dist_str}")

    # --- Save ---
    output = {
        "gpu": gpu_name,
        "model": args.model,
        "num_bins": args.num_bins,
        "h_ref": h_ref_val,
        "timing_enabled": not args.no_timing,
        "platform": platform.platform(),
        "regime_summaries": {k: {**v, "chunk_dist_aggregate": dict(v["chunk_dist_aggregate"])}
                             for k, v in regime_summaries.items()},
        "per_prompt": per_prompt_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "summary.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[hetero] Results saved to {out_path}")


if __name__ == "__main__":
    main()
