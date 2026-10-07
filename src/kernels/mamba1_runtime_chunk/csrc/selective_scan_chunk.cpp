// Host binding of the runtime-chunk Mamba-1 selective scan forward (module selective_scan_chunk_cuda).
//
// Adapted from mamba_ssm csrc/selective_scan/selective_scan.cpp (Copyright (c) 2023, Tri Dao;
// Apache License 2.0): the parameter setup and the checks of selective_scan_fwd are kept; the
// complex-weight path and the backward pass are dropped. Two entry points:
//   fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
//       the stock forward (chunk chosen from the sequence length), built from the same sources,
//   fwd_with_chunk_size(u, delta, A, B, C, D, z, delta_bias, delta_softplus, chunk_size)
//       the same kernel with the chunk as a runtime argument, chunk_size in {128, ..., 2048},
//   fwd_with_chunk_threads(u, delta, A, B, C, D, z, delta_bias, delta_softplus, chunk_size, n_threads)
//       the same kernel with the thread count fixed at n_threads (32, 64 or 128) and the chunk varied
//       through the items per thread (fp16 inputs only; review M9a).
// Both return {out, x} or {out, x, out_z}, where x holds the carried state at every chunk end
// and therefore has ceil(seqlen / chunk) chunk slots.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "selective_scan.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float) {                                    \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda_chunk(SSMParamsBase &params, int chunk, cudaStream_t stream);

template <typename input_t, typename weight_t>
void selective_scan_fwd_cuda_chunk_threads(SSMParamsBase &params, int chunk, int threads, cudaStream_t stream);

static void set_params(SSMParamsBase &params, int batch, int dim, int seqlen, int dstate, int n_groups,
                       int n_chunks, bool is_variable_B, bool is_variable_C, const at::Tensor &u,
                       const at::Tensor &delta, const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                       const at::Tensor &out, const at::Tensor &z, const at::Tensor &out_z, void *D_ptr,
                       void *delta_bias_ptr, void *x_ptr, bool has_z, bool delta_softplus) {
    memset(&params, 0, sizeof(params));
    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    params.n_chunks = n_chunks;
    params.dim_ngroups_ratio = dim / n_groups;
    params.delta_softplus = delta_softplus;
    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_bias_ptr = delta_bias_ptr;
    params.out_ptr = out.data_ptr();
    params.x_ptr = x_ptr;
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
    // All strides are in elements.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);
    if (!is_variable_B) {
        params.B_d_stride = B.stride(0);
    } else {
        params.B_batch_stride = B.stride(0);
        params.B_group_stride = B.stride(1);
    }
    params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
    if (!is_variable_C) {
        params.C_d_stride = C.stride(0);
    } else {
        params.C_batch_stride = C.stride(0);
        params.C_group_stride = C.stride(1);
    }
    params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
    params.u_batch_stride = u.stride(0);
    params.u_d_stride = u.stride(1);
    params.delta_batch_stride = delta.stride(0);
    params.delta_d_stride = delta.stride(1);
    if (has_z) {
        params.z_batch_stride = z.stride(0);
        params.z_d_stride = z.stride(1);
        params.out_z_batch_stride = out_z.stride(0);
        params.out_z_d_stride = out_z.stride(1);
    }
    params.out_batch_stride = out.stride(0);
    params.out_d_stride = out.stride(1);
}

// chunk_size == 0 selects the stock length-based dispatch; n_threads > 0 fixes the thread count.
static std::vector<at::Tensor> fwd_impl(const at::Tensor &u, const at::Tensor &delta, const at::Tensor &A,
                                        const at::Tensor &B, const at::Tensor &C,
                                        const c10::optional<at::Tensor> &D_,
                                        const c10::optional<at::Tensor> &z_,
                                        const c10::optional<at::Tensor> &delta_bias_, bool delta_softplus,
                                        int64_t chunk_size, int64_t n_threads = 0) {
    auto input_type = u.scalar_type();
    TORCH_CHECK(n_threads == 0 || (chunk_size > 0 && input_type == at::ScalarType::Half),
                "fixed-thread dispatch needs a chunk size and fp16 inputs");
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half ||
                input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(A.scalar_type() == at::ScalarType::Float, "selective_scan_chunk supports real float32 A only");
    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? at::ScalarType::Float : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? at::ScalarType::Float : input_type));
    TORCH_CHECK(u.is_cuda() && delta.is_cuda() && A.is_cuda() && B.is_cuda() && C.is_cuda());
    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int dim = sizes[1];
    const int seqlen = sizes[2];
    const int dstate = A.size(1);
    const int n_groups = is_variable_B ? B.size(1) : 1;
    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");
    CHECK_SHAPE(u, batch_size, dim, seqlen);
    CHECK_SHAPE(delta, batch_size, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (!is_variable_B) {
        CHECK_SHAPE(B, dim, dstate);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, seqlen);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    }
    if (!is_variable_C) {
        CHECK_SHAPE(C, dim, dstate);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, seqlen);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    }
    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float && D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }
    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float && delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }
    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type && z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, batch_size, dim, seqlen);
        out_z = torch::empty_like(z);
    }
    const int chunk_for_slots = chunk_size > 0 ? static_cast<int>(chunk_size) : 2048;
    const int n_chunks = (seqlen + chunk_for_slots - 1) / chunk_for_slots;
    at::Tensor out = torch::empty_like(delta);
    at::Tensor x = torch::empty({batch_size, dim, n_chunks, dstate * 2}, u.options().dtype(at::ScalarType::Float));

    SSMParamsBase params;
    set_params(params, batch_size, dim, seqlen, dstate, n_groups, n_chunks, is_variable_B, is_variable_C, u, delta, A,
               B, C, out, z, out_z, D_.has_value() ? D_.value().data_ptr() : nullptr,
               delta_bias_.has_value() ? delta_bias_.value().data_ptr() : nullptr, x.data_ptr(), has_z,
               delta_softplus);
    at::cuda::CUDAGuard device_guard{(char)u.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (n_threads > 0) {
        selective_scan_fwd_cuda_chunk_threads<at::Half, float>(params, static_cast<int>(chunk_size),
                                                               static_cast<int>(n_threads), stream);
    } else DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), "selective_scan_chunk_fwd", [&] {
        if (chunk_size > 0) {
            selective_scan_fwd_cuda_chunk<input_t, float>(params, static_cast<int>(chunk_size), stream);
        } else {
            selective_scan_fwd_cuda<input_t, float>(params, stream);
        }
    });
    std::vector<at::Tensor> result = {out, x};
    if (has_z) { result.push_back(out_z); }
    return result;
}

std::vector<at::Tensor> fwd(const at::Tensor &u, const at::Tensor &delta, const at::Tensor &A, const at::Tensor &B,
                            const at::Tensor &C, const c10::optional<at::Tensor> &D_,
                            const c10::optional<at::Tensor> &z_, const c10::optional<at::Tensor> &delta_bias_,
                            bool delta_softplus) {
    return fwd_impl(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus, 0);
}

std::vector<at::Tensor> fwd_with_chunk_size(const at::Tensor &u, const at::Tensor &delta, const at::Tensor &A,
                                            const at::Tensor &B, const at::Tensor &C,
                                            const c10::optional<at::Tensor> &D_,
                                            const c10::optional<at::Tensor> &z_,
                                            const c10::optional<at::Tensor> &delta_bias_, bool delta_softplus,
                                            int64_t chunk_size) {
    TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");
    return fwd_impl(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus, chunk_size);
}

std::vector<at::Tensor> fwd_with_chunk_threads(const at::Tensor &u, const at::Tensor &delta, const at::Tensor &A,
                                               const at::Tensor &B, const at::Tensor &C,
                                               const c10::optional<at::Tensor> &D_,
                                               const c10::optional<at::Tensor> &z_,
                                               const c10::optional<at::Tensor> &delta_bias_, bool delta_softplus,
                                               int64_t chunk_size, int64_t n_threads) {
    TORCH_CHECK(chunk_size > 0 && n_threads > 0, "chunk_size and n_threads must be positive");
    return fwd_impl(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus, chunk_size, n_threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "Selective scan forward (stock length-based chunk)");
    m.def("fwd_with_chunk_size", &fwd_with_chunk_size, "Selective scan forward with a runtime chunk size");
    m.def("fwd_with_chunk_threads", &fwd_with_chunk_threads,
          "Selective scan forward with a runtime chunk size at a fixed thread count (fp16)");
}
