
/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/kernels/cuda_graph_grouped_gemm.h"
#include "tensorrt_llm/kernels/lora/lora.h"
#include "tensorrt_llm/kernels/selectiveScan/selectiveScan.h"
#include "tensorrt_llm/thop/thUtils.h"

namespace th = torch;
namespace tk = tensorrt_llm::kernels;
using tensorrt_llm::common::fmtstr;

namespace torch_ext
{

enum class RequestType : int32_t
{
    kCONTEXT = 0,
    kGENERATION = 1
};

int64_t getNumTokens(th::Tensor const& input)
{
    int ndim = input.sizes().size();
    TLLM_CHECK_WITH_INFO(
        3 == ndim || 2 == ndim, "hidden_state dimension should be either 2 [numTokens, hidden], or 3 [b, s, hidden]");
    int64_t num_tokens = input.sizes()[0];
    if (ndim == 3)
    {
        num_tokens *= input.sizes()[1];
    }
    return num_tokens;
}

std::vector<th::Tensor> lora_grouped_gemm(th::Tensor const& input, th::Tensor const& host_request_types,
    std::vector<th::Tensor> const& lora_ranks, // numModules tensors, each tensors has single value
    std::vector<th::Tensor> const& lora_weights_pointers, th::Tensor const& host_context_lengths,
    std::vector<int64_t> const& output_hidden_sizes, bool transA, bool transB, int64_t const max_low_rank,
    int64_t const& weight_index, bool isRemoveInputPadding)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto const numReqs = lora_ranks[0].sizes()[0];
    auto const out_shape = input.sizes();
    int const numLoraModules = lora_ranks.size();
    TLLM_CHECK_WITH_INFO(lora_ranks.size() == lora_weights_pointers.size(), "both should be numLoraModules");
    std::vector<th::Tensor> output_torch;
    for (int i = 0; i < numLoraModules; i++)
    {
        std::vector<int64_t> output_shape = {out_shape[0], output_hidden_sizes[i]};
        if (!isRemoveInputPadding)
        {
            output_shape = {out_shape[0], out_shape[1], output_hidden_sizes[i]};
        }
        output_torch.push_back(torch::empty(output_shape, input.options()));
    }
    std::vector<void*> output;
    for (auto tensor_it = output_torch.begin(); tensor_it != output_torch.end(); tensor_it++)
    {
        output.push_back(tensor_it->data_ptr());
    }
    int const seqLen = isRemoveInputPadding ? 0 : input.sizes()[1];
    int32_t const* reqTypes = static_cast<int32_t const*>(host_request_types.data_ptr());
    int32_t const* hostContextLengths
        = isRemoveInputPadding ? static_cast<int32_t const*>(host_context_lengths.data_ptr()) : nullptr;

    int64_t numTokens = getNumTokens(input);

    std::vector<void const*> expandLoraWeightPtrs{};
    std::vector<int32_t> expandLoraRanks{};

    expandLoraWeightPtrs.reserve(numLoraModules * numTokens * 2);
    expandLoraRanks.reserve(numLoraModules * numTokens);

    for (int loraModuleIdx = 0; loraModuleIdx < numLoraModules; loraModuleIdx++)
    {
        auto const loraRankModule = static_cast<int32_t const*>(lora_ranks[loraModuleIdx].data_ptr());
        auto const loraWeightModulePtrs = static_cast<int64_t const*>(lora_weights_pointers[loraModuleIdx].data_ptr());

        int idx = 0;
        for (int reqId = 0; reqId < numReqs; reqId++)
        {
            // loraWeightModulePtrs has 3 pointers for each module: A,B, and an optional DoRA magnitude
            // the current LoRA plugin does not apply DoRA scaling, so the magnitude is ignored
            RequestType const reqType = static_cast<RequestType const>(reqTypes[reqId]);
            if (reqType == RequestType::kGENERATION)
            {
                expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3]));
                expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3 + 1]));
                expandLoraRanks.push_back(loraRankModule[reqId]);
                idx += 1;
            }
            else
            {
                int contextLen = (isRemoveInputPadding ? hostContextLengths[reqId] : seqLen);

                for (int contextId = 0; contextId < contextLen; contextId++)
                {
                    expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3]));
                    expandLoraWeightPtrs.push_back(reinterpret_cast<void const*>(loraWeightModulePtrs[reqId * 3 + 1]));
                    expandLoraRanks.push_back(loraRankModule[reqId]);
                    idx += 1;
                }
            }
        }

        // In 1st generation phase cross attention qkv lora, cross qkv is skipped by passing an empty encoder_output
        // (passing 0 to dim) getNumTokens() will get in cross qkv_lora. Skipping the check for this case.
        if (numTokens > 0)
        {
            TLLM_CHECK_WITH_INFO(idx == numTokens,
                fmtstr("LoraParams and input dims don't match, lora tokens %d input tokens %ld", idx, numTokens));
        }
    }

    thread_local std::shared_ptr<tensorrt_llm::common::CublasMMWrapper> cublasWrapper;
    if (cublasWrapper == nullptr)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        cublasWrapper
            = std::make_shared<tensorrt_llm::common::CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }

    int const inHiddenSize = input.sizes()[input.sizes().size() - 1];

    std::vector<int> outHiddenSizes(output_hidden_sizes.size());
    for (int i = 0; i < numLoraModules; i++)
    {
        outHiddenSizes[i] = output_hidden_sizes[i];
    }
    nvinfer1::DataType loraRuntimeDataType;
    switch (input.scalar_type())
    {
    case torch::kFloat16: loraRuntimeDataType = nvinfer1::DataType::kHALF; break;
    case torch::kBFloat16: loraRuntimeDataType = nvinfer1::DataType::kBF16; break;
    default: throw std::invalid_argument("Invalid dtype, only supports float16, bfloat16");
    }

    auto mLoraImpl = std::make_shared<tensorrt_llm::kernels::LoraImpl>(
        inHiddenSize, outHiddenSizes, transA, transB, numLoraModules, loraRuntimeDataType, max_low_rank, cublasWrapper);

    // TODO (dafrimi): use Profiler to find the best tactic as used in lora_plugin
    mLoraImpl->setBestTactic(std::nullopt);

    auto const workspace_size = mLoraImpl->getWorkspaceSize(numTokens, numReqs, loraRuntimeDataType);

    auto workspace = torch::empty(std::vector<int64_t>{static_cast<int64_t>(workspace_size)}, input.options());

    mLoraImpl->run(numTokens, numReqs, input.data_ptr(), expandLoraRanks.data(), expandLoraWeightPtrs.data(),
        weight_index, output.data(), workspace.data_ptr(), stream);
    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return output_torch;
}

th::Tensor lora_grouped_gemm_cuda_graph(th::Tensor const& input,
    th::Tensor const& lora_in_sizes,       // [layer_module_num, max_lora_size, 3]
    th::Tensor const& lora_out_sizes,      // [layer_module_num, max_lora_size, 3]
    th::Tensor const& a_offsets,           // [layer_module_num, max_lora_size]
    th::Tensor const& b_ptrs,              // [layer_module_num, max_lora_size]
    th::Tensor const& d_offsets,           // [layer_module_num, max_lora_size]
    th::Tensor const& b_prime_ptrs,        // [layer_module_num, max_lora_size]
    th::Tensor const& d_prime_offsets,     // [layer_module_num, max_lora_size]
    th::Tensor const& slot_ids,            // [batch_size] - for reference
    th::Tensor const& sorted_ids,          // [batch_size] - sorted indices for gather/scatter
    int64_t problem_count,
    th::Tensor const& intermediate_buffer, // Intermediate buffer for d
    th::Tensor const& output_buffer        // Output buffer for d'
)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto batch_size = slot_ids.sizes()[0];
    auto layer_module_num = lora_in_sizes.sizes()[0];
    auto max_lora_size = lora_in_sizes.sizes()[1];
    auto hidden_size = input.sizes()[input.sizes().size() - 1];

    // Create reordered input buffer using gather operation with sorted_ids
    // This replaces the physical reordering with efficient gather/scatter
    auto reordered_input = torch::gather(input, 0, sorted_ids.unsqueeze(1).expand({-1, input.size(1)}));

    // Create output tensor with same shape as output_buffer (not input!)
    // This is necessary because LoRA output dimensions can differ from input dimensions
    // (e.g., attention layers: input=hidden_dim, output=3*hidden_dim for q,k,v)
    auto output = torch::zeros_like(output_buffer);

    // Step 2: Convert offsets to actual pointers using CUDA Graph compatible operations
    // Use PyTorch tensor operations to add base addresses to offset tensors on GPU
    auto reordered_input_ptr = reinterpret_cast<int64_t>(reordered_input.data_ptr());
    auto intermediate_buffer_ptr = reinterpret_cast<int64_t>(intermediate_buffer.data_ptr());
    auto output_buffer_ptr = reinterpret_cast<int64_t>(output_buffer.data_ptr());

    auto typeSize = input.element_size();

    // Create actual pointer tensors by adding base addresses to offsets on GPU
    // This maintains CUDA graph compatibility by keeping operations on GPU
    // No need for .clone() since we're creating new tensors, not modifying originals
    auto a_ptrs_tensor = a_offsets * typeSize + reordered_input_ptr;
    auto d_ptrs_tensor = d_offsets * typeSize + intermediate_buffer_ptr;
    auto a_prime_ptrs_tensor = d_offsets * typeSize + intermediate_buffer_ptr; // Same as d_ptrs
    auto d_prime_ptrs_tensor = d_prime_offsets * typeSize + output_buffer_ptr;

    // Flatten tensors for grouped GEMM API
    auto a_ptrs_flat = a_ptrs_tensor.flatten();
    auto d_ptrs_flat = d_ptrs_tensor.flatten();
    auto a_prime_ptrs_flat = a_prime_ptrs_tensor.flatten();
    auto d_prime_ptrs_flat = d_prime_ptrs_tensor.flatten();

    // GPU pointers are now ready for direct use with CUDA Graph compatible GEMM functions

    // Convert flattened pointer tensors to GPU pointer arrays for CUDA Graph compatible access
    auto* a_ptrs_gpu = reinterpret_cast<void* const*>(a_ptrs_flat.data_ptr());
    auto* d_ptrs_gpu = reinterpret_cast<void* const*>(d_ptrs_flat.data_ptr());
    auto* a_prime_ptrs_gpu = reinterpret_cast<void* const*>(a_prime_ptrs_flat.data_ptr());
    auto* d_prime_ptrs_gpu = reinterpret_cast<void* const*>(d_prime_ptrs_flat.data_ptr());

    // Step 3: Prepare GEMM problem sizes using CUDA Graph compatible operations
    // The lora_in_sizes and lora_out_sizes tensors are already in cutlass::gemm::GemmCoord format
    // (3 int32 values: m, n, k) and can be cast directly to the required pointer type

    // Get GPU pointers to problem sizes tensors - these are stored in the correct format
    auto* problem_sizes_1_ptr = reinterpret_cast<cutlass::gemm::GemmCoord const*>(lora_in_sizes.data_ptr());
    auto* problem_sizes_2_ptr = reinterpret_cast<cutlass::gemm::GemmCoord const*>(lora_out_sizes.data_ptr());

    // Step 4: Get weight pointers using CUDA Graph compatible operations
    // The b_ptrs and b_prime_ptrs tensors already contain pointer values as int64
    // Cast them directly to void* arrays for the GEMM kernels

    auto* b_ptrs_gpu = reinterpret_cast<void* const*>(b_ptrs.data_ptr());
    auto* b_prime_ptrs_gpu = reinterpret_cast<void* const*>(b_prime_ptrs.data_ptr());

    // Step 5: Direct CUTLASS grouped GEMM setup (no CuBLAS wrapper needed)
    // The new CUDA Graph compatible implementation uses CUTLASS directly

    // Get data type
    nvinfer1::DataType loraRuntimeDataType;
    switch (input.scalar_type())
    {
    case torch::kFloat16: loraRuntimeDataType = nvinfer1::DataType::kHALF; break;
    case torch::kBFloat16: loraRuntimeDataType = nvinfer1::DataType::kBF16; break;
    default: throw std::invalid_argument("Invalid dtype, only supports float16, bfloat16");
    }

    // Calculate workspace size - only need execution workspace, no parameter workspace
    size_t execution_workspace_size = 8 * 1024 * 1024; // 8MB for GEMM execution
    auto execution_workspace
        = torch::empty({static_cast<int64_t>(execution_workspace_size)}, input.options().dtype(torch::kUInt8));

    // Step 6: Call CUDA Graph compatible grouped GEMM for lora_in (split-K)
    // No parameter workspace needed - tensors are already on GPU
    if (problem_count > 0)
    {
        tk::cuda_graph_splitk_grouped_gemm(problem_sizes_1_ptr, problem_count, a_ptrs_gpu, b_ptrs_gpu,
            nullptr,                        // ptrC (no bias)
            d_ptrs_gpu,
            execution_workspace.data_ptr(), // Only execution workspace needed
            execution_workspace_size,
            true,                           // isLoraIn
            loraRuntimeDataType,
            4,                              // splitKSlices
            1,                              // minKN
            stream);
    }

    // Step 7: Call CUDA Graph compatible grouped GEMM for lora_out
    if (problem_count > 0)
    {
        tk::cuda_graph_grouped_gemm(problem_sizes_2_ptr, problem_count, a_prime_ptrs_gpu, b_prime_ptrs_gpu,
            nullptr,                        // ptrC (no bias)
            d_prime_ptrs_gpu,
            execution_workspace.data_ptr(), // Only execution workspace needed
            execution_workspace_size,
            false,                          // isLoraIn
            loraRuntimeDataType,
            1,                              // minKN
            stream);
    }

    // Step 8: Reorder output back to original order using direct scatter operation
    // Since reordered_input[i] = input[sorted_ids[i]], we want output[sorted_ids[i]] = output_buffer[i]
    // This can be achieved directly with scatter using sorted_ids as indices
    output.scatter_(0, sorted_ids.unsqueeze(1).expand({-1, output_buffer.size(1)}), output_buffer);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return output;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "lora_grouped_gemm(Tensor input, "
        "Tensor host_request_types, "
        "Tensor [] lora_ranks, "
        "Tensor [] lora_weights_pointers, "
        "Tensor host_context_lengths, "
        "int [] output_hidden_sizes, "
        "bool transA, "
        "bool transB, "
        "int max_low_rank, "
        "int weight_index, "
        "bool isRemoveInputPadding) -> Tensor[]");

    m.def(
        "lora_grouped_gemm_cuda_graph(Tensor input, "
        "Tensor lora_in_sizes, "
        "Tensor lora_out_sizes, "
        "Tensor a_offsets, "
        "Tensor b_ptrs, "
        "Tensor d_offsets, "
        "Tensor b_prime_ptrs, "
        "Tensor d_prime_offsets, "
        "Tensor slot_ids, "
        "Tensor sorted_ids, "
        "int problem_count, "
        "Tensor intermediate_buffer, "
        "Tensor output_buffer) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("lora_grouped_gemm", &torch_ext::lora_grouped_gemm);
    m.impl("lora_grouped_gemm_cuda_graph", &torch_ext::lora_grouped_gemm_cuda_graph);
}
