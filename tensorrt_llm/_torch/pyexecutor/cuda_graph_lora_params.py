"""
CUDA Graph compatible LoRA parameters structure.

This module defines the new LoRA parameters structure that enables CUDA Graph capture
with multi-LoRA by using persistent device tensors for group GEMM operations.
"""

from collections import namedtuple
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

    LoraLayerKey = namedtuple('LoraLayerKey', ['layer_idx', 'module_ids'])

    @dataclass
    class LoraLayerInfo:
        module_num: int = 0
        output_sizes: List[int] | torch.Tensor | None = None
        input_hidden_size: int = 0

    def __init__(self,
                 max_batch_size: int,
                 max_lora_size: int,
                 max_rank: int,
                 layer_info: Dict[LoraLayerKey, LoraLayerInfo],
                 layer_module2key: Dict[tuple[int, int], LoraLayerKey],
                 device: str = "cuda"):
        """
        Initialize CUDA Graph compatible LoRA parameters.

        Args:
            max_batch_size: Maximum batch size for this graph
            max_lora_size: Maximum number of LoRA adapters
            max_rank: Maximum rank for all layers
            layers_info: Layer information for each layer
            device: Device to allocate tensors on
            dtype: Data type for size and offset tensors
        """
        self.max_batch_size = max_batch_size
        self.max_lora_size = max_lora_size
        self.max_rank = max_rank
        self.layer_info = layer_info
        self.layer_module2key = layer_module2key
        self.device = device

        # Sparse per-layer parameters (only for layers with LoRA modules)
        # Map: layer_idx -> LoraLayerParams (only contains layers with layer_module_nums[layer_idx] > 0)
        self.layer_params: Dict[LoraLayerKey, LoraLayerParams] = {}

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

        # persistent values for gen-only batch with cuda graph
        self.persistent_sorted_ids = self.sorted_ids

        # Initialize per-layer parameters ONLY for layers with LoRA modules
        for key, info in self.layer_info.items():
            assert info.module_num > 0 and info.output_sizes is not None and len(
                info.output_sizes
            ) == info.module_num and info.input_hidden_size >= 0
            info.output_sizes = torch.tensor(info.output_sizes)
            # Create layer parameters
            self.layer_params[key] = self._create_layer_params(
                key, info.module_num, info.output_sizes)

    def _create_layer_params(
            self, key: LoraLayerKey, layer_module_num: int,
            module_output_sizes: torch.Tensor) -> LoraLayerParams:
        """
        Create LoraLayerParams for a specific layer.

        Args:
            key: Key of the layer
            layer_module_num: Number of modules in this layer
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
            # cutlass::gemm::GemmCoord uses int32
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
            d_ld_a=torch.zeros(shape_2d, dtype=torch.int64, device=self.device),
            d_ld_b=torch.zeros(shape_2d, dtype=torch.int64, device=self.device),
            d_ld_d=torch.zeros(shape_2d, dtype=torch.int64, device=self.device),
            d_ld_b_prime=torch.zeros(shape_2d,
                                     dtype=torch.int64,
                                     device=self.device),
            d_ld_d_prime=torch.zeros(shape_2d,
                                     dtype=torch.int64,
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

        slot_ids = slot_ids[:actual_batch_size]
        slot_ids = torch.tensor(slot_ids,
                                dtype=self.persistent_sorted_ids.dtype)

        # Compute sorted indices for gather/scatter operations
        # Use stable sort to maintain deterministic ordering for tokens with same slot_id
        _, sorted_indices = torch.sort(slot_ids, stable=True)

        # Update sorted_ids tensor with the computed indices
        if actual_batch_size <= self.max_batch_size:
            # if can fit in persistent, use it
            self.sorted_ids = self.persistent_sorted_ids
            self.sorted_ids[:actual_batch_size] = sorted_indices
        else:
            # otherwise not an gen-only batch, use new allocated sorted_ids
            self.sorted_ids = sorted_indices.to(device=self.device)

    def update_weight_pointers(self, peft_table: Dict[int, List],
                               slot_to_task_mapping: Dict[int, Optional[int]]):
        """
        Update weight pointers from PEFT cache manager.

        Args:
            peft_table: PEFT table from cache manager containing weight pointers
            slot_to_task_mapping: Mapping from slot_id to task_id
        """
        host_weight_in_ptrs = dict()
        host_weight_out_ptrs = dict()
        for layer_key, layer_param in self.layer_params.items():
            host_weight_in_ptrs[layer_key] = torch.zeros_like(
                layer_param.d_b_ptrs, device='cpu')
            host_weight_out_ptrs[layer_key] = torch.zeros_like(
                layer_param.d_b_prime_ptrs, device='cpu')

        # Fill in pointers for active slots
        for slot_id, task_id in slot_to_task_mapping.items():
            # Only process actual LoRA slots (0 to max_lora_size-1), for empty slot, values have been set to 0
            if slot_id >= self.max_lora_size or task_id is None or task_id not in peft_table:
                continue

            # Get configurations for this task_id
            task_configs = peft_table[task_id]

            for config in task_configs:
                layer_id = config.layer_id
                module_id = config.module_id
                key = self.layer_module2key[(layer_id, module_id)]
                local_module_id = key.module_ids.index(module_id)

                # Only process layers that have LoRA modules
                assert key in self.layer_params, f"Layer {layer_id} not found in layer_params, assumption that all LoRA has their adapters on the same layers is broken"

                # Validate LoRA rank - TaskLayerModuleConfig uses 'rank' attribute
                rank = config.adapter_size
                assert rank <= self.max_rank, f"LoRA rank {rank} in layer {layer_id} exceeds configured max_rank {self.max_rank}. "

                # Set weight pointers for this slot and module
                host_weight_in_ptrs[key][local_module_id,
                                         slot_id] = config.weights_in_pointer
                host_weight_out_ptrs[key][local_module_id,
                                          slot_id] = config.weights_out_pointer

        for key, layer_param in self.layer_params.items():
            layer_param.d_b_ptrs.copy_(host_weight_in_ptrs[key])
            layer_param.d_b_prime_ptrs.copy_(host_weight_out_ptrs[key])

        # compute ld's
        for key, layer_param in self.layer_params.items():
            # a [bs, hidden]
            input_hidden_size = self.layer_info[key].input_hidden_size
            layer_param.d_ld_a.fill_(input_hidden_size)

            # a_prime / d [num_layer_modules, bs, max_rank]
            layer_param.d_ld_d.fill_(self.max_rank)

            # b / b_prime store as column major, so their ld should be row number
            # b [input_hidden_size, lora_rank]
            layer_param.d_ld_b.fill_(input_hidden_size)

            # d_prime [bs, sum_of_each_module_output_sizes]
            layer_param.d_ld_d_prime.fill_(
                torch.sum(self.layer_info[key].output_sizes))

    def update_gemm_sizes_and_offsets(self, batch_size: int,
                                      slot_counts: torch.Tensor,
                                      slot_to_task_mapping: Dict[int,
                                                                 Optional[int]],
                                      peft_table: Dict[int, List]):
        """
        Update GEMM sizes and buffer offsets based on current batch composition.

        Args:
            batch_size: Current batch size
            slot_counts: Number of tokens for each slot_id in the batch, shape: [max_lora_size,]
            slot_to_task_mapping: Mapping from slot_id to task_id
            peft_table: PEFT table containing adapter configurations
        """

        def get_offset_from_counts(counts: torch.Tensor) -> torch.Tensor:
            offset = torch.roll(counts, 1, dims=0)
            offset[0] = 0
            offset = torch.cumsum(offset, dim=0)
            return offset

        slot_token_offset = get_offset_from_counts(slot_counts)
        assert slot_token_offset.ndim == 1 and slot_token_offset.shape[
            0] == self.max_lora_size

        # get slot ranks
        # assume ranks are the same within a slot,
        # input_hidden_size are the same within a layer
        # output sizes are the same within a layer
        ranks = torch.zeros(self.max_lora_size, dtype=torch.int32)

        for slot_id in range(self.max_lora_size):
            task_id = slot_to_task_mapping.get(slot_id)
            if task_id is None or task_id not in peft_table:
                continue
            task_configs = peft_table[task_id]
            config = task_configs[0]
            ranks[slot_id] = config.adapter_size

        # offset shapes [num_layer_modules, max_lora_size]
        for layer_key, layer_params in self.layer_params.items():
            # reordered a [bs, hidden], each module has the same offset
            dst_shape = layer_params.d_a_offset.shape
            layer_params.d_a_offset.copy_(
                slot_token_offset.unsqueeze(0).expand(dst_shape))

            # d [num_layer_modules, bs, max_rank]
            d_offset = slot_token_offset.unsqueeze(0) + torch.arange(
                dst_shape[0]).unsqueeze(1) * self.max_lora_size
            # [num_layer_modules, max_lora_size] = [1, max_lora_size] + [num_layer_modules, 1]
            layer_params.d_d_offset.copy_(d_offset)

            # d' [bs, sum_of_each_module_output_sizes]
            bs_offset = slot_token_offset.unsqueeze(0)  # [1, max_lora_size]
            out_sizes = self.layer_info[layer_key].output_sizes
            sum_out_sizes = torch.sum(out_sizes)
            bs_offset *= sum_out_sizes

            out_offset = get_offset_from_counts(out_sizes).unsqueeze(
                1)  # [num_layer_modules, 1]

            layer_params.d_d_prime_offset.copy_(bs_offset + out_offset)

            # sizes
            input_hidden_size = torch.tensor(
                self.layer_info[layer_key].input_hidden_size).reshape(
                    (1,
                     1)).expand(dst_shape)  # [num_layer_modules, max_lora_size]
            input_hidden_size = input_hidden_size.unsqueeze(-1)

            token_counts = slot_counts.unsqueeze(0).expand(
                dst_shape)  # [num_layer_modules, max_lora_size]
            token_counts = token_counts.unsqueeze(-1)

            _ranks = ranks.unsqueeze(0).expand(
                dst_shape)  # [num_layer_modules, max_lora_size]
            _ranks = _ranks.unsqueeze(-1)

            output_sizes = self.layer_info[layer_key].output_sizes.unsqueeze(
                1).expand(dst_shape)  # [num_layer_modules, max_lora_size]
            output_sizes = output_sizes.unsqueeze(-1)

            layer_params.d_lora_in_sizes.copy_(
                torch.cat([token_counts, input_hidden_size, _ranks], dim=-1))
            layer_params.d_lora_out_sizes.copy_(
                torch.cat([token_counts, _ranks, output_sizes], dim=-1))

            # fill ldb_prime lora ranks
            # b_prime [lora_rank, module_output_size]
            layer_params.d_ld_b_prime.copy_(
                ranks.unsqueeze(0).expand(dst_shape))
            # TODO: is this correct?

            # TODO: store only one of this

    def get_problem_count(self, layer_key: LoraLayerKey) -> int:
        """
        Get the number of GEMM problems for a layer.

        Args:
            layer_key: Key of the layer

        Returns:
            Number of GEMM problems (layer_module_num * max_lora_size)
            Returns 0 if layer has no LoRA modules
            Note: Only actual LoRA slots are counted, not the dummy base model slot
        """
        if layer_key not in self.layer_params:
            return 0  # Layer has no LoRA modules
        return self.layer_info[layer_key].module_num * self.max_lora_size

    def get_layer_params(self,
                         layer_key: LoraLayerKey) -> Optional[LoraLayerParams]:
        """
        Get LoRA parameters for a specific layer.

        Args:
            layer_key: Key of the layer

        Returns:
            LoraLayerParams for the specified layer, or None if layer has no LoRA modules
        """
        return self.layer_params.get(layer_key)
