// Dynamic shared memory per block of every reference-scan instantiation (review M2b).
//
// The stock launcher (selective_scan_fwd_launch) passes
//     Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(Ktraits::scan_t)
// bytes of dynamic shared memory, which the static resource usage of the binary does not show.
// This host-only program prints that size for the instantiations of the runtime-chunk build
// (fp16 inputs, fp32 weights, kNRows = 1) as tab-separated lines:
//     threads items even_len variable_b variable_c has_z smem_bytes
// Nothing runs on the GPU. Build with the same headers as the extension, e.g.
//     nvcc -std=c++17 -I$MAMBA_CSRC -I$TORCH/include smem_sizes.cu -o smem_sizes
#include <cstdio>

#include "selective_scan_fwd_kernel.cuh"

template <int kNThreads, int kNItems, bool kIsEvenLen, bool kIsVariableB, bool kIsVariableC, bool kHasZ>
void print_one() {
    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, 1, kIsEvenLen, kIsVariableB, kIsVariableC,
                                                     kHasZ, at::Half, float>;
    constexpr int kSmemSize = Ktraits::kSmemSize + 1 * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
    std::printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n", kNThreads, kNItems, int(kIsEvenLen), int(kIsVariableB),
                int(kIsVariableC), int(kHasZ), kSmemSize);
}

template <int kNThreads, int kNItems>
void print_pair() {
    print_one<kNThreads, kNItems, false, false, false, false>();
    print_one<kNThreads, kNItems, false, false, false, true>();
    print_one<kNThreads, kNItems, false, true, true, false>();
    print_one<kNThreads, kNItems, false, true, true, true>();
    print_one<kNThreads, kNItems, true, false, false, false>();
    print_one<kNThreads, kNItems, true, false, false, true>();
    print_one<kNThreads, kNItems, true, true, true, false>();
    print_one<kNThreads, kNItems, true, true, true, true>();
}

int main() {
    // stock pairing (chunk = threads * items): 128, 256, 512, 1024, 2048
    print_pair<32, 4>();
    print_pair<32, 8>();
    print_pair<32, 16>();
    print_pair<64, 16>();
    print_pair<128, 16>();
    // fixed thread counts (review M9a): 32 threads up to 64 items, 64 threads up to 32, 128 threads up to 16
    print_pair<32, 32>();
    print_pair<32, 64>();
    print_pair<64, 4>();
    print_pair<64, 8>();
    print_pair<64, 32>();
    print_pair<128, 4>();
    print_pair<128, 8>();
    return 0;
}
