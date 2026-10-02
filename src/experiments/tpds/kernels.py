"""Recurrence-preserving chunked scan kernels used by the TPDS experiments.

Three kernel families are exposed with a real, tunable chunk (time-tile) size.
In every case one call processes the whole sequence in ONE kernel launch (or
one fixed launch pipeline) and the recurrent state is carried across chunks;
the sequence is never split into independent calls.

1. Mamba-1 selective scan (``mamba1_scan``): a Triton kernel written for this
   study. Each program owns one batch element and ``BLOCK_D`` channels and
   walks the sequence in time tiles of ``BLOCK_T`` steps. Inside a tile the
   linear recurrence h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t is solved
   with ``tl.associative_scan`` over time, one state index at a time (as in
   the reference CUDA kernel, which also loops over d_state inside a chunk);
   the state h[BLOCK_D, d_state] is carried in registers from tile to tile.
   Tunables: BLOCK_T (the chunk), BLOCK_D, num_warps, num_stages.
2. Mamba-2 SSD (``mamba2_ssd``): ``mamba_chunk_scan_combined`` from
   ``mamba_ssm`` (Triton), whose ``chunk_size`` argument is the chunk.
3. Gated DeltaNet (``gdn_chunk``): ``chunk_gated_delta_rule`` from ``fla``,
   whose ``chunk_size`` keyword accepts {16, 32, 64}.

Pure-PyTorch fp32 references are provided for all three families.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl


# =========================================================================== Mamba-1 (Triton)
@triton.jit
def _linrec_combine(a_l, b_l, a_r, b_r):
    # Composition of h -> a_l h + b_l followed by h -> a_r h + b_r.
    return a_l * a_r, a_r * b_l + b_r


@triton.jit
def _mamba1_chunked_scan_kernel(
    u_ptr, dt_ptr, A_ptr, B_ptr, C_ptr, Dskip_ptr, z_ptr, dbias_ptr, out_ptr,
    L, DIM,
    s_u_b, s_u_d, s_u_l,
    s_dt_b, s_dt_d, s_dt_l,
    s_A_d, s_A_n,
    s_B_b, s_B_n, s_B_l,
    s_C_b, s_C_n, s_C_l,
    s_z_b, s_z_d, s_z_l,
    s_o_b, s_o_d, s_o_l,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_DBIAS: tl.constexpr,
    SOFTPLUS: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < DIM
    offs_n = tl.arange(0, BLOCK_N)
    offs_tt = tl.arange(0, BLOCK_T)

    # Recurrent state for all state indices of this channel block, carried across tiles.
    h = tl.zeros([BLOCK_D, BLOCK_N], dtype=tl.float32)
    if HAS_D:
        d_skip = tl.load(Dskip_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    if HAS_DBIAS:
        d_bias = tl.load(dbias_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    u_base = u_ptr + pid_b * s_u_b + offs_d[:, None] * s_u_d
    dt_base = dt_ptr + pid_b * s_dt_b + offs_d[:, None] * s_dt_d
    B_base = B_ptr + pid_b * s_B_b
    C_base = C_ptr + pid_b * s_C_b
    o_base = out_ptr + pid_b * s_o_b + offs_d[:, None] * s_o_d

    for t0 in range(0, L, BLOCK_T):
        offs_t = t0 + offs_tt
        mask_t = offs_t < L
        m2 = mask_d[:, None] & mask_t[None, :]
        u = tl.load(u_base + offs_t[None, :] * s_u_l, mask=m2, other=0.0).to(tl.float32)
        dt = tl.load(dt_base + offs_t[None, :] * s_dt_l, mask=m2, other=0.0).to(tl.float32)
        if HAS_DBIAS:
            dt = dt + d_bias[:, None]
        if SOFTPLUS:
            dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
        dtu = dt * u
        y = tl.zeros([BLOCK_D, BLOCK_T], dtype=tl.float32)
        for n in range(N):
            a_n = tl.load(A_ptr + offs_d * s_A_d + n * s_A_n, mask=mask_d, other=0.0).to(tl.float32)
            b_n = tl.load(B_base + n * s_B_n + offs_t * s_B_l, mask=mask_t, other=0.0).to(tl.float32)
            c_n = tl.load(C_base + n * s_C_n + offs_t * s_C_l, mask=mask_t, other=0.0).to(tl.float32)
            # Masked (out-of-range) steps are identity maps: a = 1, b = 0.
            a = tl.where(m2, tl.exp(dt * a_n[:, None]), 1.0)
            bx = tl.where(m2, dtu * b_n[None, :], 0.0)
            acum, hs = tl.associative_scan((a, bx), axis=1, combine_fn=_linrec_combine)
            sel_n = offs_n[None, :] == n
            h0 = tl.sum(tl.where(sel_n, h, 0.0), axis=1)
            hs = hs + acum * h0[:, None]
            y += hs * c_n[None, :]
            h_last = tl.sum(tl.where(offs_tt[None, :] == BLOCK_T - 1, hs, 0.0), axis=1)
            h = tl.where(sel_n, h_last[:, None], h)
        if HAS_D:
            y += u * d_skip[:, None]
        if HAS_Z:
            z = tl.load(z_ptr + pid_b * s_z_b + offs_d[:, None] * s_z_d + offs_t[None, :] * s_z_l,
                        mask=m2, other=0.0).to(tl.float32)
            y = y * (z / (1.0 + tl.exp(-z)))
        tl.store(o_base + offs_t[None, :] * s_o_l, y.to(out_ptr.dtype.element_ty), mask=m2)


@dataclass(frozen=True)
class LaunchConfig:
    block_t: int
    block_d: int
    num_warps: int
    num_stages: int = 1

    def as_dict(self) -> dict:
        return {"block_t": self.block_t, "block_d": self.block_d, "num_warps": self.num_warps,
                "num_stages": self.num_stages}


# Per-chunk default launch configuration (BLOCK_D, num_warps), chosen once by a
# short offline search on the RTX 3070 at d=2048, L in {512, 2048, 8192}. It
# plays the role of a vendor default: E1/E2 vary only the chunk and take the
# launch configuration from this table; E4 searches the full joint space.
DEFAULT_MAMBA1_LAUNCH = {32: (8, 2), 64: (8, 2), 128: (8, 2), 256: (4, 2), 512: (4, 2), 1024: (4, 4)}


def default_mamba1_config(chunk: int) -> LaunchConfig:
    """Fixed launch configuration used when only the chunk is varied (E1, E2)."""
    block_d, num_warps = DEFAULT_MAMBA1_LAUNCH.get(chunk, (4, 4))
    return LaunchConfig(block_t=chunk, block_d=block_d, num_warps=num_warps, num_stages=1)


def mamba1_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
    chunk: int = 256,
    config: Optional[LaunchConfig] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mamba-1 selective scan, ``selective_scan_fn`` semantics (variable B/C, one group).

    Shapes: u, delta, z: (b, d, L); A: (d, n); B, C: (b, n, L); D, delta_bias: (d,).
    Returns y with the dtype of ``u``. The chunk is the time tile BLOCK_T.
    """
    cfg = config or default_mamba1_config(chunk)
    bsz, dim, L = u.shape
    n = A.shape[1]
    assert delta.shape == u.shape and B.shape == (bsz, n, L) and C.shape == (bsz, n, L), \
        (u.shape, delta.shape, B.shape, C.shape)
    if out is None:
        out = torch.empty_like(u)
    dummy = u
    zt = z if z is not None else dummy
    grid = (triton.cdiv(dim, cfg.block_d), bsz)
    _mamba1_chunked_scan_kernel[grid](
        u, delta, A, B, C,
        D if D is not None else dummy, zt, delta_bias if delta_bias is not None else dummy, out,
        L, dim,
        u.stride(0), u.stride(1), u.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        zt.stride(0), zt.stride(1), zt.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        N=n, BLOCK_N=triton.next_power_of_2(n), BLOCK_T=cfg.block_t, BLOCK_D=cfg.block_d,
        HAS_D=D is not None, HAS_Z=z is not None, HAS_DBIAS=delta_bias is not None,
        SOFTPLUS=delta_softplus,
        num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    return out


def ref_selective_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    """Sequential fp32 reference for the Mamba-1 selective scan (same shapes as ``mamba1_scan``)."""
    u = u.float(); delta = delta.float(); A = A.float(); B = B.float(); C = C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    bsz, dim, L = u.shape
    h = torch.zeros(bsz, dim, A.shape[1], device=u.device, dtype=torch.float32)
    ys = torch.empty(bsz, dim, L, device=u.device, dtype=torch.float32)
    for t in range(L):
        dA = torch.exp(delta[:, :, t, None] * A[None])
        h = dA * h + (delta[:, :, t] * u[:, :, t])[:, :, None] * B[:, None, :, t]
        ys[:, :, t] = (h * C[:, None, :, t]).sum(-1)
    if D is not None:
        ys = ys + u * D.float()[None, :, None]
    if z is not None:
        ys = ys * torch.nn.functional.silu(z.float())
    return ys


# =========================================================================== Mamba-2 SSD
_SSD = None


def _load_ssd():
    """Import mamba_ssm's Triton SSD ops without its compiled CUDA extension.

    ``mamba_ssm/__init__.py`` imports ``selective_scan_cuda``, which does not
    exist when the package is installed with MAMBA_SKIP_CUDA_BUILD=TRUE. The
    Triton SSD modules do not need it, so a namespace stub for the top-level
    package is registered and only ``mamba_ssm.ops.triton.ssd_combined`` is
    imported. A full install (with the CUDA extension) is used as-is.
    """
    global _SSD
    if _SSD is not None:
        return _SSD
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore
    except Exception:
        spec = importlib.util.find_spec("mamba_ssm")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("mamba_ssm is not installed (see src/experiments/tpds/README.md)")
        for k in [k for k in sys.modules if k == "mamba_ssm" or k.startswith("mamba_ssm.")]:
            del sys.modules[k]
        stub = types.ModuleType("mamba_ssm")
        stub.__path__ = list(spec.submodule_search_locations)
        stub.__version__ = "stub"
        sys.modules["mamba_ssm"] = stub
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # type: ignore
    _SSD = mamba_chunk_scan_combined
    return _SSD


def ssd_available() -> bool:
    try:
        _load_ssd()
        return True
    except Exception:
        return False


def mamba2_ssd(x, dt, A, B, C, chunk_size: int, D=None, dt_bias=None, dt_softplus=True, z=None):
    """Mamba-2 SSD forward. x: (b, L, h, p); dt: (b, L, h); A: (h,); B, C: (b, L, g, n)."""
    fn = _load_ssd()
    return fn(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus)


def ref_ssd(x, dt, A, B, C, D=None, dt_bias=None, dt_softplus=True):
    """fp32 reference for SSD, expressed as a Mamba-1 selective scan with per-head scalar decay."""
    bsz, L, h, p = x.shape
    g, n = B.shape[2], B.shape[3]
    rep = h // g
    y = torch.empty(bsz, L, h, p, device=x.device, dtype=torch.float32)
    for gi in range(g):
        heads = range(gi * rep, (gi + 1) * rep)
        for hi in heads:
            u = x[:, :, hi, :].permute(0, 2, 1).float()  # (b, p, L)
            dlt = dt[:, :, hi].float()
            if dt_bias is not None:
                dlt = dlt + dt_bias[hi].float()
            dlt = dlt[:, None, :].expand(bsz, p, L)
            Ad = A[hi].float().expand(p, n)
            Bh = B[:, :, gi, :].permute(0, 2, 1).float()
            Ch = C[:, :, gi, :].permute(0, 2, 1).float()
            Dh = None if D is None else D[hi].float().expand(p)
            y[:, :, hi, :] = ref_selective_scan(u, dlt, Ad, Bh, Ch, Dh, None, None, dt_softplus).permute(0, 2, 1)
    return y


# =========================================================================== Gated DeltaNet
_GDN = None


def _load_gdn():
    global _GDN
    if _GDN is None:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # type: ignore

        _GDN = chunk_gated_delta_rule
    return _GDN


def gdn_available() -> bool:
    try:
        _load_gdn()
        return True
    except Exception:
        return False


GDN_CHUNK_EXPOSED = True  # fla 0.5.x accepts chunk_size in {16, 32, 64} via **kwargs


def gdn_chunk(q, k, v, g, beta, chunk_size: int = 64):
    """Gated DeltaNet chunked forward. q, k: (b, T, H, K); v: (b, T, H, V); g, beta: (b, T, H)."""
    fn = _load_gdn()
    o, _ = fn(q, k, v, g, beta, chunk_size=chunk_size)
    return o


def gdn_chunk_prod(q, k, v, g, beta, A_log, dt_bias, chunk_size: int = 64, allow_neg_eigval: bool = False):
    """Gated DeltaNet chunked forward as the FLA layer of a checkpoint calls it (production path).

    q, k, v: (b, T, H, K / V) raw projections (q and k are L2-normalized inside the kernel); g: (b, T, H),
    the output of ``a_proj``, turned into the log-gate inside the kernel with ``A_log`` and ``dt_bias``;
    beta: (b, T, H), the output of ``b_proj``, passed through the sigmoid inside the kernel. These are the
    tensors ``capture_gdn_traces.py`` stores.
    """
    fn = _load_gdn()
    o, _ = fn(q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
              use_beta_sigmoid_in_kernel=True, allow_neg_eigval=allow_neg_eigval, state_v_first=True,
              chunk_size=chunk_size)
    return o


def ref_gdn(q, k, v, g, beta):
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule  # type: ignore

    o, _ = naive_recurrent_gated_delta_rule(q.float(), k.float(), v.float(), beta.float(), g.float())
    return o


# =========================================================================== synthetic inputs
def make_mamba1_inputs(bsz, dim, n, L, dtype=torch.float16, device="cuda", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    u = torch.randn(bsz, dim, L, device=device, dtype=torch.float32, generator=gen)
    # Pre-softplus delta around softplus^-1(0.01..0.1), the Mamba init range.
    delta = torch.randn(bsz, dim, L, device=device, generator=gen) * 0.5 - 3.5
    A = -torch.exp(torch.rand(dim, n, device=device, generator=gen) * math.log(16.0))
    B = torch.randn(bsz, n, L, device=device, generator=gen)
    C = torch.randn(bsz, n, L, device=device, generator=gen)
    D = torch.randn(dim, device=device, generator=gen)
    z = torch.randn(bsz, dim, L, device=device, generator=gen)
    return {"u": u.to(dtype), "delta": delta.to(dtype), "A": A.float().contiguous(), "B": B.to(dtype),
            "C": C.to(dtype), "D": D.float(), "z": z.to(dtype)}


def make_ssd_inputs(bsz, L, nheads=32, headdim=64, ngroups=1, dstate=128, dtype=torch.float16, device="cuda", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(bsz, L, nheads, headdim, device=device, generator=gen)
    dt = torch.randn(bsz, L, nheads, device=device, generator=gen) * 0.5 - 3.5
    A = -torch.exp(torch.rand(nheads, device=device, generator=gen) * math.log(16.0))
    B = torch.randn(bsz, L, ngroups, dstate, device=device, generator=gen)
    C = torch.randn(bsz, L, ngroups, dstate, device=device, generator=gen)
    D = torch.randn(nheads, device=device, generator=gen)
    dt_bias = torch.randn(nheads, device=device, generator=gen) * 0.1
    return {"x": x.to(dtype), "dt": dt.to(dtype), "A": A.float(), "B": B.to(dtype), "C": C.to(dtype),
            "D": D.float(), "dt_bias": dt_bias.float()}


def make_gdn_inputs(bsz, L, nheads=4, dk=256, dv=512, dtype=torch.bfloat16, device="cuda", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(bsz, L, nheads, dk, device=device, generator=gen)
    k = torch.nn.functional.normalize(torch.randn(bsz, L, nheads, dk, device=device, generator=gen), p=2, dim=-1)
    v = torch.randn(bsz, L, nheads, dv, device=device, generator=gen)
    g = torch.nn.functional.logsigmoid(torch.randn(bsz, L, nheads, device=device, generator=gen) + 3.0)
    beta = torch.sigmoid(torch.randn(bsz, L, nheads, device=device, generator=gen))
    return {"q": q.to(dtype), "k": k.to(dtype), "v": v.to(dtype), "g": g.float(), "beta": beta.to(dtype)}


# =========================================================================== validation
def _err(y, ref):
    y = y.float(); ref = ref.float()
    diff = (y - ref).abs()
    scale = ref.abs().max().clamp(min=1e-6)
    return {"max_abs_err": float(diff.max()), "max_rel_err": float(diff.max() / scale),
            "mean_abs_err": float(diff.mean()), "ref_max_abs": float(scale)}


def validate_mamba1(chunks, L=300, dim=48, n=16, bsz=2, dtype=torch.float32, tol=None, with_z=True, device="cuda"):
    """Check the Triton kernel against the fp32 reference for each chunk (L chosen not a multiple of any chunk)."""
    inp = make_mamba1_inputs(bsz, dim, n, L, dtype=dtype, device=device, seed=1)
    z = inp["z"] if with_z else None
    ref = ref_selective_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], z)
    tol = tol if tol is not None else (1e-4 if dtype == torch.float32 else 2e-2)
    rows = []
    for c in chunks:
        y = mamba1_scan(inp["u"], inp["delta"], inp["A"], inp["B"], inp["C"], inp["D"], z, chunk=c)
        e = _err(y, ref)
        e.update({"family": "mamba1", "chunk": c, "L": L, "dim": dim, "n": n, "batch": bsz,
                  "dtype": str(dtype).replace("torch.", ""), "tol_rel": tol, "passed": e["max_rel_err"] <= tol})
        rows.append(e)
    return rows


def validate_ssd(chunks, L=300, nheads=4, headdim=16, dstate=32, bsz=2, dtype=torch.float32, tol=None, device="cuda"):
    inp = make_ssd_inputs(bsz, L, nheads, headdim, 1, dstate, dtype=dtype, device=device, seed=2)
    ref = ref_ssd(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], inp["D"], inp["dt_bias"])
    tol = tol if tol is not None else (5e-3 if dtype == torch.float32 else 3e-2)
    rows = []
    for c in chunks:
        y = mamba2_ssd(inp["x"], inp["dt"], inp["A"], inp["B"], inp["C"], c, D=inp["D"], dt_bias=inp["dt_bias"])
        e = _err(y, ref)
        e.update({"family": "mamba2_ssd", "chunk": c, "L": L, "nheads": nheads, "headdim": headdim,
                  "dstate": dstate, "batch": bsz, "dtype": str(dtype).replace("torch.", ""),
                  "tol_rel": tol, "passed": e["max_rel_err"] <= tol})
        rows.append(e)
    return rows


def validate_gdn(chunks, L=300, nheads=2, dk=64, dv=64, bsz=1, dtype=torch.bfloat16, tol=5e-2, device="cuda"):
    inp = make_gdn_inputs(bsz, L, nheads, dk, dv, dtype=dtype, device=device, seed=3)
    ref = ref_gdn(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"])
    rows = []
    for c in chunks:
        y = gdn_chunk(inp["q"], inp["k"], inp["v"], inp["g"], inp["beta"], chunk_size=c)
        e = _err(y, ref)
        e.update({"family": "gdn", "chunk": c, "L": L, "nheads": nheads, "dk": dk, "dv": dv, "batch": bsz,
                  "dtype": str(dtype).replace("torch.", ""), "tol_rel": tol, "passed": e["max_rel_err"] <= tol})
        rows.append(e)
    return rows


if __name__ == "__main__":  # quick self-test
    import json

    from src.experiments.tpds.common import GDN_CHUNKS, MAMBA1_CHUNKS, MAMBA2_CHUNKS

    out = validate_mamba1(MAMBA1_CHUNKS) + validate_mamba1(MAMBA1_CHUNKS, dtype=torch.float16, L=777)
    if ssd_available():
        out += validate_ssd(MAMBA2_CHUNKS)
    if gdn_available():
        out += validate_gdn(GDN_CHUNKS)
    for r in out:
        print(json.dumps(r))
    print("ALL PASSED" if all(r["passed"] for r in out) else "SOME FAILED")
