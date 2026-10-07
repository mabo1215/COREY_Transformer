// Fixed-thread-count instantiations (review M9a), fp16 inputs only: the captured calls are fp16.
#include "selective_scan_fwd_chunk.cuh"

template void selective_scan_fwd_cuda_chunk_threads<at::Half, float>(SSMParamsBase &params, int chunk, int threads,
                                                                      cudaStream_t stream);
