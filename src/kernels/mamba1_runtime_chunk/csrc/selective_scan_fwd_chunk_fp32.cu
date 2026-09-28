// Split into one file per input type so that the three instantiations compile in parallel.
#include "selective_scan_fwd_chunk.cuh"

template void selective_scan_fwd_cuda_chunk<float, float>(SSMParamsBase &params, int chunk, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, float>(SSMParamsBase &params, cudaStream_t stream);
