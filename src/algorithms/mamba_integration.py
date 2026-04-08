from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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


class MambaBackend(ABC):
    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        raise NotImplementedError

    @abstractmethod
    def score_perplexity(self, text: str) -> float:
        raise NotImplementedError


class HuggingFaceMambaBackend(MambaBackend):
    def __init__(
        self,
        model_spec: ModelSpec,
        runtime_config: RuntimeConfig,
        scheduler_hook: EntropyGuidedSchedulerHook | None = None,
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

    def _load_awq_model(self, model_kwargs: dict[str, Any]) -> Any:
        try:
            from awq import AutoAWQForCausalLM
        except ImportError as exc:
            raise ImportError(
                "AWQ loading requires the autoawq package."
            ) from exc

        quantization = self.runtime_config.quantization
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
        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ImportError as exc:
            raise ImportError(
                "GPTQ loading requires the auto-gptq package."
            ) from exc

        quantization = self.runtime_config.quantization
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
        if quantization_mode == "fp16":
            model_kwargs["torch_dtype"] = getattr(torch, self.runtime_config.dtype)
        else:
            model_kwargs["torch_dtype"] = getattr(torch, "float16")

        self.tokenizer = auto_tokenizer.from_pretrained(self.model_spec.model_id, trust_remote_code=True)
        if quantization_backend in {"awq", "autoawq"}:
            self.model = self._load_awq_model(model_kwargs)
        elif quantization_backend in {"gptq", "autogptq", "auto-gptq"}:
            self.model = self._load_gptq_model(model_kwargs)
        else:
            self.model = auto_model.from_pretrained(self.model_spec.model_id, **model_kwargs)
        self.model.to(self.runtime_config.device)
        self.model.eval()

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None or self.torch is None:
            self.load()

    def generate(self, request: GenerationRequest) -> GenerationOutput:
        self._ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.torch is not None

        encoded = self.tokenizer(request.prompt, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self.runtime_config.device) for key, value in encoded.items()}

        entropy_before: float | None = None
        entropy_after: float | None = None
        tile_size: int | None = None
        with self.torch.no_grad():
            if self.scheduler_hook is not None:
                embeddings = self.model.get_input_embeddings()(encoded["input_ids"])
                profile = self.scheduler_hook.profile_hidden_states(embeddings)
                entropy_before = float(profile["entropy"])
                entropy_after = entropy_before
                tile_size = int(profile["tile_size"])

            start_time = time.perf_counter()
            generated = self.model.generate(
                **encoded,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
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

    def score_perplexity(self, text: str) -> float:
        self._ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.torch is not None

        encoded = self.tokenizer(text, return_tensors="pt", truncation=True)
        encoded = {key: value.to(self.runtime_config.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded, labels=encoded["input_ids"])
        return float(self.torch.exp(outputs.loss).item())


def default_mamba_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(name="mamba-370m", model_id="state-spaces/mamba-370m-hf"),
        ModelSpec(name="mamba-1.4b", model_id="state-spaces/mamba-1.4b-hf"),
        ModelSpec(name="mamba-2.8b", model_id="state-spaces/mamba-2.8b-hf"),
    ]