from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any


def official_mamba_fast_path_status() -> dict[str, bool | str | None]:
    status: dict[str, bool | str | None] = {
        "selective_scan_fn": None,
        "mamba_inner_fn": None,
        "selective_state_update": None,
        "causal_conv1d_fn": None,
        "causal_conv1d_update": None,
        "error": None,
    }
    try:
        from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    except Exception as exc:  # pragma: no cover - depends on optional CUDA extensions
        status["error"] = repr(exc)
        return status

    status["selective_scan_fn"] = selective_scan_fn is not None
    status["mamba_inner_fn"] = mamba_inner_fn is not None

    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update

        status["selective_state_update"] = selective_state_update is not None
    except Exception:
        status["selective_state_update"] = False

    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

        status["causal_conv1d_fn"] = causal_conv1d_fn is not None
        status["causal_conv1d_update"] = causal_conv1d_update is not None
    except Exception:
        status["causal_conv1d_fn"] = False
        status["causal_conv1d_update"] = False
    return status


@dataclass(frozen=True)
class QuantizationConfig:
    mode: str = "fp16"
    backend: str | None = None
    bits: int | None = None
    group_size: int | None = 128
    use_exllama: bool = False
    fuse_layers: bool = False


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str | None = None
    trust_remote_code: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda"
    dtype: str = "float16"
    max_length: int = 8192
    batch_size: int = 1
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeTelemetry:
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float
    tokens_per_second: float
    dram_bytes_per_token: float | None = None
    energy_j_per_token: float | None = None


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    telemetry: RuntimeTelemetry
    entropy_before: float | None = None
    entropy_after: float | None = None
    suggested_tile_size: int | None = None


class EntropyGuidedSchedulerHook:
    def __init__(self, entropy_threshold: float = 5.0, bins: int = 256) -> None:
        self.entropy_threshold = entropy_threshold
        self.bins = bins

    def profile_hidden_states(self, hidden_states: Any) -> dict[str, float | int | bool]:
        try:
            from .torch_entropy import activation_entropy, recommend_tile_size
        except ImportError as exc:
            raise ImportError(
                "EntropyGuidedSchedulerHook requires torch and src.algorithms.torch_entropy."
            ) from exc

        entropy = float(activation_entropy(hidden_states, num_bins=self.bins).item())
        tile_size = recommend_tile_size(entropy)
        return {
            "entropy": entropy,
            "tile_size": tile_size,
            "should_fuse": entropy >= self.entropy_threshold,
        }


class StaticTileSchedulerHook:
    def __init__(self, tile_size: int = 256) -> None:
        self.tile_size = tile_size

    def profile_hidden_states(self, hidden_states: Any) -> dict[str, float | int | bool | None]:
        # Static policy intentionally emits a fixed tile recommendation without entropy computation.
        return {
            "entropy": None,
            "tile_size": int(self.tile_size),
            "should_fuse": True,
        }


class MambaBackend(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        raise NotImplementedError

    @abstractmethod
    def score_perplexity(self, text: str) -> float | None:
        raise NotImplementedError

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        return [self.generate(request) for request in requests]

    def score_perplexity_batch(self, texts: list[str]) -> list[float | None]:
        return [self.score_perplexity(text) for text in texts]


class HuggingFaceMambaBackend(MambaBackend):
    def __init__(
        self,
        model_spec: ModelSpec,
        runtime_config: RuntimeConfig,
        scheduler_hook: Any | None = None,
    ) -> None:
        self.model_spec = model_spec
        self.runtime_config = runtime_config
        self.scheduler_hook = scheduler_hook
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.torch: Any | None = None

    def _require_dependencies(self) -> tuple[Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "mamba_integration requires torch and transformers for real checkpoint loading."
            ) from exc
        return torch, AutoModelForCausalLM, AutoTokenizer

    def _transformers_uses_dtype_keyword(self) -> bool:
        version = self._package_version("transformers")
        if version is None:
            return False
        major_version = int(version.split(".", maxsplit=1)[0])
        return major_version >= 5

    def _package_version(self, package_name: str) -> str | None:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return None

    def _load_awq_model(self, model_kwargs: dict[str, Any]) -> Any:
        try:
            from awq import AutoAWQForCausalLM
        except ImportError as exc:
            raise ImportError(
                f"AWQ loading requires a working autoawq stack; import failed with: {exc}"
            ) from exc

        if "mamba" in self.model_spec.model_id.lower():
            raise RuntimeError(
                "AutoAWQ does not currently support Mamba checkpoints in the verified WSL2 stack: "
                "a direct probe with autoawq 0.2.9 in adama-cuda128 returned `mamba isn't supported yet`. "
                "Revisit this path only after upstream Mamba support lands."
            )

        quantization = self.runtime_config.quantization
        awq_version = self._package_version("autoawq") or self._package_version("awq")
        if awq_version is None:
            raise ImportError("Unable to determine the installed autoawq/awq version.")
        load_kwargs = {
            "trust_remote_code": self.model_spec.trust_remote_code,
            "fuse_layers": quantization.fuse_layers,
            "safetensors": True,
        }
        if self.model_spec.revision is not None:
            load_kwargs["revision"] = self.model_spec.revision
        if quantization.group_size is not None:
            load_kwargs["group_size"] = quantization.group_size
        return AutoAWQForCausalLM.from_quantized(self.model_spec.model_id, **load_kwargs)

    def _load_gptq_model(self, model_kwargs: dict[str, Any]) -> Any:
        if "mamba" in self.model_spec.model_id.lower():
            raise RuntimeError(
                "GPTQ loading for Mamba is currently unavailable in the verified WSL2 stack. "
                "auto-gptq 0.7.1 fails against the active transformers API before model loading, "
                "and upstream Mamba support is not yet available. Revisit this path only after "
                "pinning a known-compatible GPTQ stack with explicit Mamba support."
            )

        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ImportError as exc:
            raise ImportError(
                f"GPTQ loading requires a working auto-gptq stack; import failed with: {exc}"
            ) from exc

        quantization = self.runtime_config.quantization
        gptq_version = self._package_version("auto-gptq") or self._package_version("auto_gptq")
        if gptq_version is None:
            raise ImportError("Unable to determine the installed auto-gptq version.")
        load_kwargs = {
            "device": self.runtime_config.device,
            "trust_remote_code": self.model_spec.trust_remote_code,
            "use_safetensors": True,
            "use_exllama": quantization.use_exllama,
        }
        if self.model_spec.revision is not None:
            load_kwargs["revision"] = self.model_spec.revision
        if quantization.bits is not None:
            load_kwargs["bits"] = quantization.bits
        if quantization.group_size is not None:
            load_kwargs["group_size"] = quantization.group_size
        return AutoGPTQForCausalLM.from_quantized(self.model_spec.model_id, **load_kwargs)

    def _maybe_move_to_device(self) -> None:
        assert self.model is not None
        quantization_backend = (self.runtime_config.quantization.backend or "").lower()
        if quantization_backend and quantization_backend not in {"none", "fp16"}:
            return
        if hasattr(self.model, "to"):
            self.model.to(self.runtime_config.device)

    def _prepare_quantized_model(self) -> None:
        assert self.model is not None
        if hasattr(self.model, "eval"):
            self.model.eval()

    def load(self) -> None:
        torch, auto_model, auto_tokenizer = self._require_dependencies()
        self.torch = torch

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.model_spec.trust_remote_code,
        }
        if self.model_spec.revision is not None:
            model_kwargs["revision"] = self.model_spec.revision

        quantization_mode = self.runtime_config.quantization.mode.lower()
        quantization_backend = (self.runtime_config.quantization.backend or "").lower()
        dtype_key = "dtype" if self._transformers_uses_dtype_keyword() else "torch_dtype"
        if quantization_mode in {"fp16", "fp32"}:
            model_kwargs[dtype_key] = getattr(torch, self.runtime_config.dtype)
        else:
            model_kwargs[dtype_key] = getattr(torch, "float16")

        try:
            self.tokenizer = auto_tokenizer.from_pretrained(
                self.model_spec.model_id, trust_remote_code=True
            )
        except Exception as e_fast:
            print(f"[WARN] Fast tokenizer load failed: {e_fast}. Trying slow tokenizer fallback.")
            try:
                self.tokenizer = auto_tokenizer.from_pretrained(
                    self.model_spec.model_id, trust_remote_code=True, use_fast=False
                )
            except Exception as e_slow:
                raise RuntimeError(
                    f"Both fast and slow tokenizer loading failed for model '{self.model_spec.model_id}'. "
                    f"Fast error: {e_fast}\nSlow error: {e_slow}\n"
                    "This usually means the model repository is missing required tokenizer files (e.g., tokenizer.json, spiece.model, vocab.txt, merges.txt) or does not specify a compatible tokenizer class. "
                    "Please check the model card on HuggingFace, try a different model, or contact the model author to add the necessary tokenizer files. "
                    "If you know the correct tokenizer class, you can manually instantiate it in the code."
                )
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"
        if getattr(self.tokenizer, "pad_token_id", None) is None and getattr(self.tokenizer, "eos_token_id", None) is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if quantization_backend in {"awq", "autoawq"}:
            self.model = self._load_awq_model(model_kwargs)
        elif quantization_backend in {"gptq", "autogptq", "auto-gptq"}:
            self.model = self._load_gptq_model(model_kwargs)
        else:
            self.model = auto_model.from_pretrained(self.model_spec.model_id, **model_kwargs)
        self._maybe_move_to_device()
        self._prepare_quantized_model()

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self.torch is None:
            self.load()

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self._ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.torch is not None

        encoded = self.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.runtime_config.max_length,
        )
        encoded = {key: value.to(self.runtime_config.device) for key, value in encoded.items()}

        entropy_before: float | None = None
        entropy_after: float | None = None
        tile_size: int | None = None
        with self.torch.no_grad():
            if self.scheduler_hook is not None:
                embeddings = self.model.get_input_embeddings()(encoded["input_ids"])
                profile = self.scheduler_hook.profile_hidden_states(embeddings)
                profile_entropy = profile.get("entropy")
                if profile_entropy is not None:
                    entropy_before = float(profile_entropy)
                    entropy_after = entropy_before
                profile_tile_size = profile.get("tile_size")
                if profile_tile_size is not None:
                    tile_size = int(profile_tile_size)

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "do_sample": request.do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if request.do_sample:
                generation_kwargs["temperature"] = request.temperature
                generation_kwargs["top_p"] = request.top_p

            start_time = time.perf_counter()
            generated = self.model.generate(
                **encoded,
                **generation_kwargs,
            )
            elapsed_s = time.perf_counter() - start_time

        output_tokens = generated[0, encoded["input_ids"].shape[-1] :]
        generated_text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        generated_token_count = int(output_tokens.shape[-1])
        latency_ms = elapsed_s * 1000.0
        tokens_per_second = generated_token_count / elapsed_s if elapsed_s > 0 else 0.0
        telemetry = RuntimeTelemetry(
            prompt_tokens=int(encoded["input_ids"].shape[-1]),
            generated_tokens=generated_token_count,
            latency_ms=latency_ms,
            tokens_per_second=tokens_per_second,
        )
        return GenerationOutput(
            text=generated_text,
            telemetry=telemetry,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            suggested_tile_size=tile_size,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        self._ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.torch is not None
        if not requests:
            return []

        prompts = [request.prompt for request in requests]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.runtime_config.max_length,
        )
        encoded = {key: value.to(self.runtime_config.device) for key, value in encoded.items()}

        max_new_tokens = max(request.max_new_tokens for request in requests)
        do_sample = any(request.do_sample for request in requests)
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = max(request.temperature for request in requests)
            generation_kwargs["top_p"] = min(request.top_p for request in requests)

        entropy_before_values: list[float | None] = [None] * len(requests)
        tile_sizes: list[int | None] = [None] * len(requests)
        with self.torch.no_grad():
            if self.scheduler_hook is not None:
                embeddings = self.model.get_input_embeddings()(encoded["input_ids"])
                for index in range(embeddings.shape[0]):
                    profile = self.scheduler_hook.profile_hidden_states(embeddings[index : index + 1])
                    profile_entropy = profile.get("entropy")
                    if profile_entropy is not None:
                        entropy_before_values[index] = float(profile_entropy)
                    profile_tile_size = profile.get("tile_size")
                    if profile_tile_size is not None:
                        tile_sizes[index] = int(profile_tile_size)

            start_time = time.perf_counter()
            generated = self.model.generate(
                **encoded,
                **generation_kwargs,
            )
            elapsed_s = time.perf_counter() - start_time

        attention_mask = encoded.get("attention_mask")
        outputs: list[GenerationOutput] = []
        for index, request in enumerate(requests):
            prompt_length = int(attention_mask[index].sum().item()) if attention_mask is not None else int(encoded["input_ids"].shape[1])
            generated_tokens = generated[index, prompt_length : prompt_length + request.max_new_tokens]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_token_count = int(generated_tokens.shape[-1])
            telemetry = RuntimeTelemetry(
                prompt_tokens=prompt_length,
                generated_tokens=generated_token_count,
                latency_ms=(elapsed_s * 1000.0) / len(requests),
                tokens_per_second=(generated_token_count / elapsed_s) if elapsed_s > 0 else 0.0,
            )
            outputs.append(
                GenerationOutput(
                    text=generated_text,
                    telemetry=telemetry,
                    entropy_before=entropy_before_values[index],
                    entropy_after=entropy_before_values[index],
                    suggested_tile_size=tile_sizes[index],
                )
            )
        return outputs

    def score_perplexity(self, text: str) -> float | None:
        self._ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.torch is not None

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.runtime_config.max_length,
        )
        encoded = {key: value.to(self.runtime_config.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded, labels=encoded["input_ids"])
        return float(self.torch.exp(outputs.loss).item())

    def score_perplexity_batch(self, texts: list[str]) -> list[float | None]:
        return [self.score_perplexity(text) for text in texts]


class OllamaBackend(MambaBackend):
    def __init__(
        self,
        model_spec: ModelSpec,
        runtime_config: RuntimeConfig,
        scheduler_hook: Any | None = None,
        host: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model_spec = model_spec
        self.runtime_config = runtime_config
        self.scheduler_hook = scheduler_hook
        self.host = host.rstrip("/")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=f"{self.host}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach Ollama at {self.host}") from exc
        return json.loads(body)

    def load(self) -> None:
        self._post_json("/api/show", {"model": self.model_spec.model_id})

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        start_time = time.perf_counter()
        response = self._post_json(
            "/api/generate",
            {
                "model": self.model_spec.model_id,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_new_tokens,
                },
            },
        )
        elapsed_s = time.perf_counter() - start_time
        prompt_tokens = int(response.get("prompt_eval_count", 0))
        generated_tokens = int(response.get("eval_count", 0))
        total_duration = response.get("total_duration")
        latency_ms = float(total_duration) / 1_000_000.0 if total_duration is not None else elapsed_s * 1000.0
        eval_duration = response.get("eval_duration")
        if eval_duration:
            tokens_per_second = generated_tokens / (float(eval_duration) / 1_000_000_000.0)
        else:
            tokens_per_second = generated_tokens / elapsed_s if elapsed_s > 0 else 0.0
        return GenerationOutput(
            text=str(response.get("response", "")),
            telemetry=RuntimeTelemetry(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                latency_ms=latency_ms,
                tokens_per_second=tokens_per_second,
            ),
        )

    def score_perplexity(self, text: str) -> float | None:
        return None


def default_mamba_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(name="mamba-370m", model_id="state-spaces/mamba-370m-hf"),
        ModelSpec(name="mamba-1.4b", model_id="state-spaces/mamba-1.4b-hf"),
        ModelSpec(name="mamba-2.7b", model_id="benchang1110/mamba2-2.7b-hf"),
        ModelSpec(name="pythia-410m", model_id="EleutherAI/pythia-410m", trust_remote_code=False),
        ModelSpec(name="pythia-1.4b", model_id="EleutherAI/pythia-1.4b", trust_remote_code=False),
        ModelSpec(name="pythia-2.8b", model_id="EleutherAI/pythia-2.8b", trust_remote_code=False),
    ]