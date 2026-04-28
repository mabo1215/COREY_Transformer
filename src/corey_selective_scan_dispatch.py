"""Configurable COREY selective-scan dispatch module.

This module is intentionally conservative.  It exposes a
``selective_scan_fn(..., chunk_size=...)`` entrypoint so the existing H800
integration harness can load it with:

    --selective-scan-dispatch-module src.corey_selective_scan_dispatch

The default backend is ``auto``: use a patched runtime-chunk CUDA extension if
one is installed, otherwise fall back to the stock mamba_ssm kernel and raise a
TypeError when a routed ``chunk_size`` is requested.  That preserves the
original path instead of silently claiming a routed speedup.

Backends:
  - auto: patched CUDA when available, otherwise original fallback.
  - original: stock mamba_ssm selective_scan_fn, no chunk routing.
  - chunked_emulation: split the sequence and call the stock kernel per chunk.
    This is for debugging only because it resets recurrence state at chunk
    boundaries; probe scripts mark it ineligible for W1 speedup evidence.
  - torch_ref: full PyTorch reference scan, correctness fallback only.
"""

from __future__ import annotations

import os
from typing import Any


VALID_CHUNKS = (32, 64, 128, 256, 512, 1024, 2048)


def _backend_name() -> str:
    return os.environ.get("COREY_SELECTIVE_SCAN_BACKEND", "auto").strip().lower() or "auto"


def _original_selective_scan_fn() -> Any:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    return selective_scan_fn


def _torch_ref_selective_scan_fn() -> Any:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

    return selective_scan_ref


def _runtime_cuda_support() -> tuple[bool, str | None]:
    """Detect a patched extension that accepts runtime chunk_size.

    The upstream mamba_ssm extension exports ``selective_scan_cuda.fwd`` only,
    with a hard-coded 2048-token chunk in C++.  A patched build may export one
    of these names.  The wrapper does not provide the CUDA patch by itself; it
    simply uses it when present.
    """

    try:
        import torch  # noqa: F401
        import selective_scan_cuda  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on remote CUDA env
        return False, f"selective_scan_cuda import failed: {exc!r}"

    for name in ("fwd_with_chunk_size", "fwd_chunked", "fwd_runtime_chunk"):
        if hasattr(selective_scan_cuda, name):
            return True, name
    return False, "patched fwd_with_chunk_size/fwd_chunked/fwd_runtime_chunk not found"


def _slice_sequence_param(tensor: Any, start: int, end: int, *, complex_pair: bool = False) -> Any:
    if tensor is None:
        return None
    if tensor.dim() < 3:
        return tensor
    if complex_pair:
        return tensor[..., start * 2 : end * 2]
    return tensor[..., start:end]


def _call_original(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any = None,
    z: Any = None,
    delta_bias: Any = None,
    *,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Any:
    return _original_selective_scan_fn()(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


def _call_torch_ref(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any = None,
    z: Any = None,
    delta_bias: Any = None,
    *,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Any:
    return _torch_ref_selective_scan_fn()(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


def _call_chunked_emulation(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any = None,
    z: Any = None,
    delta_bias: Any = None,
    *,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    chunk_size: int,
) -> Any:
    import torch

    if chunk_size not in VALID_CHUNKS:
        raise ValueError(f"chunk_size={chunk_size} not in supported set {VALID_CHUNKS}")

    seq_len = int(u.shape[-1])
    outs: list[Any] = []
    last_state = None
    complex_pair = bool(getattr(A, "is_complex", lambda: False)())
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        result = _call_original(
            u[..., start:end],
            delta[..., start:end],
            A,
            _slice_sequence_param(B, start, end, complex_pair=complex_pair),
            _slice_sequence_param(C, start, end, complex_pair=complex_pair),
            D=D,
            z=_slice_sequence_param(z, start, end, complex_pair=False),
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=True,
        )
        out, last_state = result if isinstance(result, tuple) else (result, None)
        outs.append(out)

    merged = torch.cat(outs, dim=-1)
    return (merged, last_state) if return_last_state else merged


def _call_runtime_cuda(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any = None,
    z: Any = None,
    delta_bias: Any = None,
    *,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    chunk_size: int,
) -> Any:
    """Call a patched CUDA extension if the H800 image provides one.

    This intentionally supports several candidate symbol names so the Python
    dispatch layer can remain stable while the C++/CUDA extension evolves.
    """

    if chunk_size not in VALID_CHUNKS:
        raise ValueError(f"chunk_size={chunk_size} not in supported set {VALID_CHUNKS}")

    import torch
    from einops import rearrange
    import selective_scan_cuda  # type: ignore

    fwd = None
    for name in ("fwd_with_chunk_size", "fwd_chunked", "fwd_runtime_chunk"):
        fwd = getattr(selective_scan_cuda, name, None)
        if fwd is not None:
            break
    if fwd is None:
        raise RuntimeError("patched selective_scan_cuda runtime-chunk forward is unavailable")

    squeeze_B = False
    squeeze_C = False
    if u.stride(-1) != 1:
        u = u.contiguous()
    if delta.stride(-1) != 1:
        delta = delta.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    if B.dim() == 3:
        B = rearrange(B, "b dstate l -> b 1 dstate l")
        squeeze_B = True
    if C.dim() == 3:
        C = rearrange(C, "b dstate l -> b 1 dstate l")
        squeeze_C = True

    try:
        out, x, *rest = fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus, int(chunk_size))
    except TypeError:
        out, x, *rest = fwd(
            u,
            delta,
            A,
            B,
            C,
            D,
            z,
            delta_bias,
            delta_softplus,
            chunk_size=int(chunk_size),
        )
    _ = squeeze_B, squeeze_C  # kept for parity with upstream layout handling
    last_state = x[:, :, -1, 1::2]
    if z is not None:
        out = rest[0]
    return (out, last_state) if return_last_state else out


def selective_scan_fn(
    u: Any,
    delta: Any,
    A: Any,
    B: Any,
    C: Any,
    D: Any = None,
    z: Any = None,
    delta_bias: Any = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    *,
    chunk_size: int | None = None,
) -> Any:
    """selective_scan_fn-compatible dispatch entrypoint."""

    backend = _backend_name()
    runtime_ok, runtime_reason = _runtime_cuda_support()
    if backend == "auto":
        backend = "runtime_cuda" if runtime_ok else "original"

    if backend == "runtime_cuda":
        if not runtime_ok:
            fallback = os.environ.get("COREY_SELECTIVE_SCAN_FALLBACK", "").strip().lower()
            if fallback == "original":
                backend = "original"
            else:
                raise RuntimeError(f"runtime CUDA chunk backend unavailable: {runtime_reason}")
        elif chunk_size is None:
            return _call_original(
                u,
                delta,
                A,
                B,
                C,
                D=D,
                z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=return_last_state,
            )
        else:
            return _call_runtime_cuda(
                u,
                delta,
                A,
                B,
                C,
                D=D,
                z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=return_last_state,
                chunk_size=chunk_size,
            )

    if backend == "chunked_emulation":
        if chunk_size is None:
            return _call_original(
                u,
                delta,
                A,
                B,
                C,
                D=D,
                z=z,
                delta_bias=delta_bias,
                delta_softplus=delta_softplus,
                return_last_state=return_last_state,
            )
        return _call_chunked_emulation(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
            chunk_size=chunk_size,
        )

    if backend == "torch_ref":
        if chunk_size is not None:
            raise TypeError("torch_ref backend preserves recurrence but does not honor chunk_size")
        return _call_torch_ref(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )

    if backend == "original":
        if chunk_size is not None:
            raise TypeError("original backend does not honor runtime chunk_size")
        return _call_original(
            u,
            delta,
            A,
            B,
            C,
            D=D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )

    raise ValueError(
        "Unknown COREY_SELECTIVE_SCAN_BACKEND="
        f"{backend!r}; expected auto/original/runtime_cuda/chunked_emulation/torch_ref"
    )


def get_dispatch_info() -> dict[str, Any]:
    requested = _backend_name()
    runtime_ok, runtime_reason = _runtime_cuda_support()
    if requested == "auto":
        resolved = "runtime_cuda" if runtime_ok else "original"
    elif requested == "runtime_cuda" and not runtime_ok:
        fallback = os.environ.get("COREY_SELECTIVE_SCAN_FALLBACK", "").strip().lower()
        resolved = "original" if fallback == "original" else "runtime_cuda_unavailable"
    else:
        resolved = requested
    return {
        "module": __name__,
        "requested_backend": requested,
        "resolved_backend": resolved,
        "runtime_cuda_available": runtime_ok,
        "runtime_cuda_reason": runtime_reason,
        "valid_chunks": list(VALID_CHUNKS),
        "chunk_size_honored": resolved in {"runtime_cuda", "chunked_emulation"},
        "recurrence_preserving": resolved in {"runtime_cuda", "original", "torch_ref"},
        "eligible_for_w1_speedup": resolved == "runtime_cuda",
        "debug_only": resolved in {"chunked_emulation", "torch_ref", "original"},
        "fallback_hint": (
            "Set COREY_SELECTIVE_SCAN_BACKEND=original to force the stock path, "
            "or COREY_SELECTIVE_SCAN_BACKEND=chunked_emulation for debug-only "
            "chunk routing that is not recurrence-preserving."
        ),
    }
