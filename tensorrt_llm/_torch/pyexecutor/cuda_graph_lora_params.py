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
    # Weight pointers managed by PEFT cache manager
    # Shape: [layer_module_num, max_lora_size]
    d_b_ptrs: torch.Tensor  # Lora_in weight pointers
    d_b_prime_ptrs: torch.Tensor  # Lora_out weight pointers


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

        def is_enabled(self) -> bool:
            """
            Check if the layer is enabled in sample LoRA task. We assume that all LoRA tasks occupy the same layers. TODO: remove this restriction?
            Only input_hidden_size is initialized by the sample LoRA task, so use that to check if the layer is enabled.
            """
            return self.input_hidden_size > 0

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

        # TODO: to be removed
        self.slot_ids = torch.empty(1, dtype=torch.int32, device=device)

        # Sorted indices for gather/scatter operations (persistent device tensor)
        # Will contain indices that sort tokens by their slot_ids for efficient gather/scatter
        # Shape: [max_batch_size], values are indices in [0, max_batch_size-1]
        self.sorted_ids = torch.zeros(max_batch_size,
                                      dtype=torch.int64,
                                      device=device)

        # persistent values for gen-only batch with cuda graph
        self.persistent_sorted_ids = self.sorted_ids

        self.slot_counts = torch.zeros(max_lora_size,
                                       dtype=torch.int32,
                                       device=device)
        self.slot_offsets = torch.zeros(max_lora_size,
                                        dtype=torch.int64,
                                        device=device)
        self.slot_ranks = torch.zeros(max_lora_size,
                                      dtype=torch.int32,
                                      device=device)

        # Initialize per-layer parameters ONLY for layers with LoRA modules
        for key, info in self.layer_info.items():
            if not info.is_enabled():
                continue
            assert info.module_num > 0 and info.output_sizes is not None and len(
                info.output_sizes) == info.module_num
            # Create layer parameters
            self.layer_params[key] = self._create_layer_params(
                key, info.module_num, info.output_sizes)
        print(f"layer_info: {self.layer_info}")

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

        return LoraLayerParams(
            # Weight pointers - managed by PEFT cache manager
            d_b_ptrs=torch.zeros(shape_2d,
                                 dtype=torch.int64,
                                 device=self.device),
            d_b_prime_ptrs=torch.zeros(shape_2d,
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

        self.slot_ranks.copy_(ranks)

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

    @staticmethod
    def get_offset_from_counts(counts: torch.Tensor) -> torch.Tensor:
        offset = torch.empty_like(counts, dtype=torch.int64)
        offset[0] = 0
        offset[1:] = counts[:-1]
        offset[1:].cumsum_(dim=0)
        return offset

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

        slot_token_offset = self.get_offset_from_counts(slot_counts)
        assert slot_token_offset.ndim == 1 and slot_token_offset.shape[
            0] == self.max_lora_size

        self.slot_counts.copy_(slot_counts)
        self.slot_offsets.copy_(slot_token_offset)

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
