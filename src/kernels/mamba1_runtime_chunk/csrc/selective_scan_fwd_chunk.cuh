// Runtime-chunk dispatch for the reference Mamba-1 selective scan forward kernel.
//
// The stock kernel (mamba_ssm, csrc/selective_scan/selective_scan_fwd_kernel.cuh) walks the
// sequence in chunks of kChunkSize = kNThreads * kNItems, both compile-time constants, and the
// stock host code picks (kNThreads, kNItems) from the sequence length (one chunk of <= 2048
// steps). Here the chunk is a runtime argument: every supported chunk uses the same
// (kNThreads, kNItems) pair that the stock code uses for a sequence of that length, and the
// host code sets n_chunks = ceil(seqlen / chunk). The kernel itself is unchanged.
#pragma once

#include "selective_scan_fwd_kernel.cuh"

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda_chunk(SSMParamsBase &params, int chunk, cudaStream_t stream) {
    switch (chunk) {
        case 128:  selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream); break;
        case 256:  selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream); break;
        case 512:  selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream); break;
        case 1024: selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream); break;
        case 2048: selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream); break;
        default: TORCH_CHECK(false, "selective_scan_chunk: unsupported chunk ", chunk,
                             " (supported: 128, 256, 512, 1024, 2048)");
    }
}

// Fixed thread count (review M9a). The stock pairing above changes the number of threads per block
// together with the chunk (32 threads up to chunk 512, then 64 and 128), so a chunk sweep with it
// also varies the intra-block parallelism. Here the thread count is held fixed and only the items
// per thread change: chunk = threads * kNItems, with kNItems a multiple of 4 (stock constraint).
//   32 threads: chunks 128, 256, 512, 1024, 2048 (kNItems 4 ... 64)
//   64 threads: chunks 256, 512, 1024, 2048      (kNItems 4 ... 32)
//  128 threads: chunks 512, 1024, 2048           (kNItems 4 ... 16)
template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda_chunk_threads(SSMParamsBase &params, int chunk, int threads, cudaStream_t stream) {
    const int key = threads * 4096 + chunk;
    switch (key) {
        case 32 * 4096 + 128:   selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream); break;
        case 32 * 4096 + 256:   selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream); break;
        case 32 * 4096 + 512:   selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream); break;
        case 32 * 4096 + 1024:  selective_scan_fwd_launch<32, 32, input_t, weight_t>(params, stream); break;
        case 32 * 4096 + 2048:  selective_scan_fwd_launch<32, 64, input_t, weight_t>(params, stream); break;
        case 64 * 4096 + 256:   selective_scan_fwd_launch<64, 4, input_t, weight_t>(params, stream); break;
        case 64 * 4096 + 512:   selective_scan_fwd_launch<64, 8, input_t, weight_t>(params, stream); break;
        case 64 * 4096 + 1024:  selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream); break;
        case 64 * 4096 + 2048:  selective_scan_fwd_launch<64, 32, input_t, weight_t>(params, stream); break;
        case 128 * 4096 + 512:  selective_scan_fwd_launch<128, 4, input_t, weight_t>(params, stream); break;
        case 128 * 4096 + 1024: selective_scan_fwd_launch<128, 8, input_t, weight_t>(params, stream); break;
        case 128 * 4096 + 2048: selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream); break;
        default: TORCH_CHECK(false, "selective_scan_chunk: unsupported (threads, chunk) = (", threads, ", ", chunk,
                             "); threads 32: chunks 128-2048, 64: 256-2048, 128: 512-2048");
    }
}
