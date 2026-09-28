# Runtime-chunk reference Mamba-1 selective scan (P1)

The reference CUDA selective scan of `mamba_ssm` fixes its chunk at compile time
(`kChunkSize = kNThreads * kNItems`) and chooses it from the sequence length: 128,
256, 512 and 1024 steps for sequences up to that length, 2048 above. The routed
end-to-end study of the manuscript used a patched build with the chunk as a runtime
argument; that build existed only on the H800 host, which is no longer reachable,
and its source was not kept. This directory re-implements it.

What changes: nothing inside the kernel. `csrc/selective_scan_fwd_chunk.cuh`
dispatches a runtime chunk to the same `(kNThreads, kNItems)` pair that the stock
code uses for a sequence of that length, and the host code sets
`n_chunks = ceil(seqlen / chunk)` (and sizes the saved per-chunk state `x`
accordingly). So chunk `C` on a sequence of length `L > C` runs exactly the kernel
that the stock code runs on a sequence of length `C`, for `ceil(L / C)` chunks.

| chunk | kNThreads | kNItems |
|------:|----------:|--------:|
| 128   | 32        | 4       |
| 256   | 32        | 8       |
| 512   | 32        | 16      |
| 1024  | 64        | 16      |
| 2048  | 128       | 16      |

Module `selective_scan_chunk_cuda`:
- `fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)`: stock length-based
  dispatch, compiled from the same sources (the same-binary baseline);
- `fwd_with_chunk_size(..., delta_softplus, chunk_size)`: the runtime chunk; this is
  the name `src/corey_selective_scan_dispatch.py` looks for.

Both return `[out, x]` or `[out, x, out_z]`. Only the forward pass and real float32
weights are built.

## Build (on the GPU host; no GPU needed for compilation)

```bash
# mamba_ssm source matching the installed wheel (2.3.2.post1 on vgpu3090)
source /etc/network_turbo   # AutoDL hosts
git clone --depth 1 --branch v2.3.2.post1 https://github.com/state-spaces/mamba.git /root/autodl-tmp/mamba_src
cd src/kernels/mamba1_runtime_chunk
MAMBA_CSRC=/root/autodl-tmp/mamba_src/csrc/selective_scan TORCH_CUDA_ARCH_LIST=8.6 \
    pip install --no-build-isolation -v .
```

## Validation

`python -m src.experiments.tpds.e8_refscan --validate-only` compares every chunk
against the installed stock `selective_scan_cuda.fwd` on lengths that are not a
multiple of any chunk (max |dy| and the relative error, fp16 inputs).
