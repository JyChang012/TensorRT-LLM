"""
CUDA Graph compatible LoRA parameters structure.

This module defines the new LoRA parameters structure that enables CUDA Graph capture
with multi-LoRA by using persistent device tensors for group GEMM operations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class LoraLayerParams:
    """
    Parameters for a single LoRA layer that are compatible with CUDA Graph capture.

    All tensors are persistent device tensors that can be updated outside of graph replay.
    """
    # GEMM problem sizes for lora_in and lora_out
    # Shape: [layer_module_num, max_lora_size, 3] where 3 = {m, n, k}
    d_lora_in_sizes: torch.Tensor
    d_lora_out_sizes: torch.Tensor

    # Input and output offsets for buffer addressing
    # Shape: [layer_module_num, max_lora_size]
    d_a_offset: torch.Tensor  # Layer input offsets
    d_d_offset: torch.Tensor  # Lora_in output / lora_out input offsets
    d_d_prime_offset: torch.Tensor  # Lora layer output offsets

    # Weight pointers managed by PEFT cache manager
    # Shape: [layer_module_num, max_lora_size]
    d_b_ptrs: torch.Tensor  # Lora_in weight pointers
    d_b_prime_ptrs: torch.Tensor  # Lora_out weight pointers

    # Leading dimensions for tensors
    # Shape: [layer_module_num, max_lora_size]
    d_ld_a: torch.Tensor  # Leading dimension for input A
    d_ld_b: torch.Tensor  # Leading dimension for lora_in weights B
    d_ld_d: torch.Tensor  # Leading dimension for intermediate output D
    d_ld_b_prime: torch.Tensor  # Leading dimension for lora_out weights B'
    d_ld_d_prime: torch.Tensor  # Leading dimension for final output D'

    # Base addresses for intermediate buffers (set during graph capture)
    intermediate_buffer_base: Optional[
        torch.Tensor] = None  # Base address for d buffer
    output_buffer_base: Optional[
        torch.Tensor] = None  # Base address for d' buffer

    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        expected_shape_2d = self.d_lora_in_sizes.shape[:
                                                       2]  # [layer_module_num, max_lora_size]
        expected_shape_3d = self.d_lora_in_sizes.shape  # [layer_module_num, max_lora_size, 3]

        # Validate 3D tensors
        assert self.d_lora_out_sizes.shape == expected_shape_3d, \
            f"d_lora_out_sizes shape mismatch: {self.d_lora_out_sizes.shape} vs {expected_shape_3d}"

        # Validate 2D tensors
        tensors_2d = [
            self.d_a_offset, self.d_d_offset, self.d_d_prime_offset,
            self.d_b_ptrs, self.d_b_prime_ptrs, self.d_ld_a, self.d_ld_b,
            self.d_ld_d, self.d_ld_b_prime, self.d_ld_d_prime
        ]

        for tensor in tensors_2d:
            assert tensor.shape == expected_shape_2d, \
                f"Tensor shape mismatch: {tensor.shape} vs {expected_shape_2d}"


class CudaGraphLoraParams:
    """
    CUDA Graph compatible LoRA parameters for all layers and batch management.

    This structure maintains persistent device tensors that can be updated outside
    of CUDA Graph replay to support different LoRA combinations per batch.
    """

    def __init__(self,
                 num_layers: int,
                 max_batch_size: int,
                 max_lora_size: int,
                 layer_module_nums: List[int],
                 max_ranks: List[int],
                 output_sizes: List[List[int]],
                 device: str = "cuda",
                 dtype: torch.dtype = torch.int32):
        """
        Initialize CUDA Graph compatible LoRA parameters.

        Args:
            num_layers: Number of model layers
            max_batch_size: Maximum batch size for this graph
            max_lora_size: Maximum number of LoRA adapters
            layer_module_nums: Number of modules per layer (e.g., [3, 3, ...] for q,k,v per layer)
            max_ranks: Maximum rank for each layer
            output_sizes: Output sizes for each module in each layer
            device: Device to allocate tensors on
            dtype: Data type for size and offset tensors
        """
        self.num_layers = num_layers
        self.max_batch_size = max_batch_size
        self.max_lora_size = max_lora_size
        self.layer_module_nums = layer_module_nums
        self.device = device
        self.dtype = dtype

        # Per-layer parameters
        self.layer_params: Dict[int, LoraLayerParams] = {}

        # Global slot assignment for the batch
        # Shape: [max_batch_size], values in [0, max_lora_size] where max_lora_size is for base model
        self.slot_ids = torch.zeros(max_batch_size,
                                    dtype=torch.int32,
                                    device=device)

        # Sorted indices for gather/scatter operations (persistent device tensor)
        # Will contain indices that sort tokens by their slot_ids for efficient gather/scatter
        # Shape: [max_batch_size], values are indices in [0, max_batch_size-1]
        self.sorted_ids = torch.zeros(max_batch_size,
                                      dtype=torch.int64,
                                      device=device)

        # Initialize per-layer parameters
        for layer_idx in range(num_layers):
            self.layer_params[layer_idx] = self._create_layer_params(
                layer_idx, layer_module_nums[layer_idx], max_ranks[layer_idx],
                output_sizes[layer_idx]
                if layer_idx < len(output_sizes) else [])

    def _create_layer_params(self, layer_idx: int, layer_module_num: int,
                             max_rank: int,
                             module_output_sizes: List[int]) -> LoraLayerParams:
        """
        Create LoraLayerParams for a specific layer.

        Args:
            layer_idx: Index of the layer
            layer_module_num: Number of modules in this layer
            max_rank: Maximum rank for this layer
            module_output_sizes: Output sizes for each module in this layer

        Returns:
            LoraLayerParams for the specified layer
        """
        # GEMM parameter tensors only need max_lora_size (no dummy slot for base model)
        # Base model requests are handled separately and don't participate in GEMM operations
        shape_2d = (layer_module_num, self.max_lora_size)
        shape_3d = (layer_module_num, self.max_lora_size, 3)

        return LoraLayerParams(
            # GEMM sizes - stored in cutlass::gemm::GemmCoord format (3 int32 values: m, n, k)
            # This allows direct GPU pointer casting without .item() calls
            d_lora_in_sizes=torch.zeros(shape_3d,
                                        dtype=torch.int32,
                                        device=self.device),
            d_lora_out_sizes=torch.zeros(shape_3d,
                                         dtype=torch.int32,
                                         device=self.device),

            # Offsets - will be computed based on input layout and slot assignments
            d_a_offset=torch.zeros(shape_2d,
                                   dtype=torch.int64,
                                   device=self.device),
            d_d_offset=torch.zeros(shape_2d,
                                   dtype=torch.int64,
                                   device=self.device),
            d_d_prime_offset=torch.zeros(shape_2d,
                                         dtype=torch.int64,
                                         device=self.device),

            # Weight pointers - managed by PEFT cache manager
            d_b_ptrs=torch.zeros(shape_2d,
                                 dtype=torch.int64,
                                 device=self.device),
            d_b_prime_ptrs=torch.zeros(shape_2d,
                                       dtype=torch.int64,
                                       device=self.device),

            # Leading dimensions - use int32 to match GEMM API expectations
            d_ld_a=torch.zeros(shape_2d, dtype=torch.int32, device=self.device),
            d_ld_b=torch.zeros(shape_2d, dtype=torch.int32, device=self.device),
            d_ld_d=torch.zeros(shape_2d, dtype=torch.int32, device=self.device),
            d_ld_b_prime=torch.zeros(shape_2d,
                                     dtype=torch.int32,
                                     device=self.device),
            d_ld_d_prime=torch.zeros(shape_2d,
                                     dtype=torch.int32,
                                     device=self.device),
        )

    def update_slot_ids(self, slot_ids: List[int], actual_batch_size: int):
        """
        Update slot IDs for the current batch and compute sorted indices.

        Args:
            slot_ids: List of slot IDs for each token in the batch
            actual_batch_size: Actual batch size (may be less than max_batch_size)
        """
        assert actual_batch_size <= self.max_batch_size, \
            f"Actual batch size {actual_batch_size} exceeds max {self.max_batch_size}"

        # Update slot_ids tensor (values can be 0 to max_lora_size, where max_lora_size = base model)
        self.slot_ids[:actual_batch_size] = torch.tensor(
            slot_ids[:actual_batch_size], dtype=torch.int32, device=self.device)

        # Zero out unused positions (use 0 as default, which is a valid LoRA slot)
        if actual_batch_size < self.max_batch_size:
            self.slot_ids[actual_batch_size:] = 0

        # Compute sorted indices for gather/scatter operations
        # Use stable sort to maintain deterministic ordering for tokens with same slot_id
        _, sorted_indices = torch.sort(self.slot_ids[:actual_batch_size],
                                       stable=True)

        # Update sorted_ids tensor with the computed indices
        self.sorted_ids[:actual_batch_size] = sorted_indices

        # Fill remaining positions with sequential indices (won't be used but maintains valid indices)
        if actual_batch_size < self.max_batch_size:
            self.sorted_ids[actual_batch_size:] = torch.arange(
                actual_batch_size,
                self.max_batch_size,
                dtype=torch.int64,
                device=self.device)

    def update_weight_pointers(self, peft_table: "PeftTable",
                               slot_to_task_mapping: Dict[int, Optional[int]]):
        """
        Update weight pointers from PEFT cache manager.

        Args:
            peft_table: PEFT table from cache manager containing weight pointers
            slot_to_task_mapping: Mapping from slot_id to task_id
        """
        # Reset all pointers to null (0)
        for layer_idx in range(self.num_layers):
            layer_params = self.layer_params[layer_idx]
            layer_params.d_b_ptrs.zero_()
            layer_params.d_b_prime_ptrs.zero_()

        # Fill in pointers for active slots
        for slot_id, task_id in slot_to_task_mapping.items():
            # Only process actual LoRA slots (0 to max_lora_size-1)
            if slot_id >= self.max_lora_size or task_id is None or task_id not in peft_table:
                continue

            # Get configurations for this task_id
            task_configs = peft_table[task_id]

            for config in task_configs:
                layer_id = config.layer_id
                module_id = config.module_id

                if layer_id < self.num_layers:
                    layer_params = self.layer_params[layer_id]

                    # Set weight pointers for this slot and module
                    if hasattr(config, 'in_pointer') and hasattr(
                            config, 'out_pointer'):
                        layer_params.d_b_ptrs[module_id,
                                              slot_id] = config.in_pointer
                        layer_params.d_b_prime_ptrs[
                            module_id, slot_id] = config.out_pointer

    def update_gemm_sizes_and_offsets(self, batch_size: int,
                                      input_hidden_size: int,
                                      slot_counts: Dict[int, int],
                                      slot_to_task_mapping: Dict[int,
                                                                 Optional[int]],
                                      peft_table: "PeftTable"):
        """
        Update GEMM sizes and buffer offsets based on current batch composition.

        Args:
            batch_size: Current batch size
            input_hidden_size: Hidden size of input tensors
            slot_counts: Number of tokens for each slot_id in the batch
            slot_to_task_mapping: Mapping from slot_id to task_id
            peft_table: PEFT table containing adapter configurations
        """
        for layer_idx in range(self.num_layers):
            layer_params = self.layer_params[layer_idx]
            layer_module_num = self.layer_module_nums[layer_idx]

            # Reset sizes and offsets
            layer_params.d_lora_in_sizes.zero_()
            layer_params.d_lora_out_sizes.zero_()
            layer_params.d_a_offset.zero_()
            layer_params.d_d_offset.zero_()
            layer_params.d_d_prime_offset.zero_()

            current_input_offset = 0
            current_intermediate_offset = 0
            current_output_offset = 0

            for slot_id in range(self.max_lora_size
                                 ):  # Only actual LoRA slots, no dummy slot
                task_id = slot_to_task_mapping.get(slot_id)
                token_count = slot_counts.get(slot_id, 0)

                if task_id is None or task_id not in peft_table or token_count == 0:
                    continue

                # Find configurations for this layer and task
                task_configs = peft_table[task_id]
                layer_configs = [
                    cfg for cfg in task_configs if cfg.layer_id == layer_idx
                ]

                for module_id, config in enumerate(layer_configs):
                    if module_id >= layer_module_num:
                        break

                    rank = getattr(config, 'rank', 0)
                    output_size = getattr(config, 'out_dim', input_hidden_size)

                    # Set GEMM sizes for lora_in: [M, N, K] = [token_count, input_hidden_size, rank]
                    layer_params.d_lora_in_sizes[module_id, slot_id,
                                                 0] = token_count  # M
                    layer_params.d_lora_in_sizes[module_id, slot_id,
                                                 1] = input_hidden_size  # N
                    layer_params.d_lora_in_sizes[module_id, slot_id,
                                                 2] = rank  # K

                    # Set GEMM sizes for lora_out: [M, N, K] = [token_count, rank, output_size]
                    layer_params.d_lora_out_sizes[module_id, slot_id,
                                                  0] = token_count  # M
                    layer_params.d_lora_out_sizes[module_id, slot_id,
                                                  1] = rank  # N
                    layer_params.d_lora_out_sizes[module_id, slot_id,
                                                  2] = output_size  # K

                    # Set offsets (will be converted to actual addresses later)
                    layer_params.d_a_offset[module_id,
                                            slot_id] = current_input_offset
                    layer_params.d_d_offset[
                        module_id, slot_id] = current_intermediate_offset
                    layer_params.d_d_prime_offset[
                        module_id, slot_id] = current_output_offset

                # Update offsets for next slot
                current_input_offset += token_count * input_hidden_size
                # Intermediate buffer size depends on max rank across modules
                max_rank = max(
                    [getattr(cfg, 'rank', 0) for cfg in layer_configs] + [0])
                current_intermediate_offset += token_count * max_rank
                # Output buffer size depends on sum of output sizes across modules
                total_output_size = sum([
                    getattr(cfg, 'out_dim', input_hidden_size)
                    for cfg in layer_configs
                ])
                current_output_offset += token_count * total_output_size

    def get_problem_count(self, layer_idx: int) -> int:
        """
        Get the number of GEMM problems for a layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Number of GEMM problems (layer_module_num * max_lora_size)
            Note: Only actual LoRA slots are counted, not the dummy base model slot
        """
        return self.layer_module_nums[layer_idx] * self.max_lora_size

    def get_layer_params(self, layer_idx: int) -> LoraLayerParams:
        """
        Get LoRA parameters for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            LoraLayerParams for the specified layer
        """
        return self.layer_params[layer_idx]
