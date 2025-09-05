/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "cutlass/gemm/gemm_coord.h"
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

/**
 * @brief CUDA Graph compatible wrapper for grouped GEMM operations.
 *
 * This function accepts GPU pointers directly without any workspace for parameters,
 * making it fully compatible with CUDA Graph capture and replay.
 *
 * @param problem_sizes_ptr GPU pointer to array of cutlass::gemm::GemmCoord
 * @param problem_count Number of GEMM problems
 * @param ptrA_gpu GPU pointer to array of A matrix pointers
 * @param ptrB_gpu GPU pointer to array of B matrix pointers
 * @param ptrC_gpu GPU pointer to array of C matrix pointers (can be nullptr)
 * @param ptrD_gpu GPU pointer to array of D matrix pointers
 * @param gemmExecutionWorkspace GPU workspace for GEMM execution only
 * @param gemmExecutionWorkspaceSize Size of GEMM execution workspace
 * @param isLoraIn Whether this is for LoRA input transformation
 * @param dataType Data type of the matrices
 * @param minKN Minimum K*N value for kernel selection
 * @param stream CUDA stream
 */
void cuda_graph_grouped_gemm(cutlass::gemm::GemmCoord const* problem_sizes_ptr, int problem_count,
    void* const* ptrA_gpu, void* const* ptrB_gpu, void* const* ptrC_gpu, void* const* ptrD_gpu,
    void* gemmExecutionWorkspace, int64_t gemmExecutionWorkspaceSize, bool isLoraIn, nvinfer1::DataType dataType,
    int minKN, cudaStream_t stream);

/**
 * @brief CUDA Graph compatible wrapper for split-K grouped GEMM operations.
 *
 * Similar to cuda_graph_grouped_gemm but uses split-K algorithm for better
 * performance with certain problem sizes. No parameter workspace needed.
 */
void cuda_graph_splitk_grouped_gemm(cutlass::gemm::GemmCoord const* problem_sizes_ptr, int problem_count,
    void* const* ptrA_gpu, void* const* ptrB_gpu, void* const* ptrC_gpu, void* const* ptrD_gpu,
    void* gemmExecutionWorkspace, int64_t gemmExecutionWorkspaceSize, bool isLoraIn, nvinfer1::DataType dataType,
    int splitKSlices, int minKN, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
