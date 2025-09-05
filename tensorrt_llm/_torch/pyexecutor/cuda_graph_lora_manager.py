"""
CUDA Graph compatible LoRA manager for PyTorch backend.

This module provides a manager that coordinates AdapterSlotManager, CudaGraphLoraParams,
and PeftCacheManager to enable CUDA Graph capture with multi-LoRA support.
"""

from collections import Counter
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from .adapter_slot_manager import AdapterSlotManager
from .cuda_graph_lora_params import CudaGraphLoraParams

if TYPE_CHECKING:
    from tensorrt_llm.batch_manager.peftCacheManager import PeftTable

    from ..attention_backend.interface import AttentionMetadata
    from .scheduler import ScheduledRequests


class CudaGraphLoraManager:
    """
    Manager that coordinates adapter slots and CUDA Graph compatible LoRA parameters.

    This class bridges the gap between the current LoRA implementation and the new
    CUDA Graph compatible design by managing adapter slots and preparing persistent
    device tensors for group GEMM operations.
    """

    def __init__(self,
                 num_layers: int,
                 max_lora_size: int,
                 layer_module_nums: List[int],
                 max_ranks: List[int],
                 output_sizes: List[List[int]],
                 device: str = "cuda"):
        """
        Initialize the CUDA Graph LoRA manager.

        Args:
            num_layers: Number of model layers
            max_lora_size: Maximum number of LoRA adapters that can be active
            layer_module_nums: Number of modules per layer (e.g., [3, 3, ...] for q,k,v)
            max_ranks: Maximum rank for each layer
            output_sizes: Output sizes for each module in each layer
            device: Device to allocate tensors on
        """
        self.num_layers = num_layers
        self.max_lora_size = max_lora_size
        self.device = device

        self.adapter_slot_manager = AdapterSlotManager(max_lora_size, device)

        # Store lora_params for different batch sizes
        self.batch_size_to_lora_params: Dict[int, CudaGraphLoraParams] = {}

        # Configuration for creating lora_params
        self.layer_module_nums = layer_module_nums
        self.max_ranks = max_ranks
        self.output_sizes = output_sizes

    def get_or_create_lora_params(self, batch_size: int) -> CudaGraphLoraParams:
        """
        Get or create CudaGraphLoraParams for a specific batch size.

        Args:
            batch_size: The batch size to get/create params for

        Returns:
            CudaGraphLoraParams for the specified batch size
        """
        if batch_size not in self.batch_size_to_lora_params:
            self.batch_size_to_lora_params[batch_size] = CudaGraphLoraParams(
                num_layers=self.num_layers,
                max_batch_size=batch_size,
                max_lora_size=self.max_lora_size,
                layer_module_nums=self.layer_module_nums,
                max_ranks=self.max_ranks,
                output_sizes=self.output_sizes,
                device=self.device)
        return self.batch_size_to_lora_params[batch_size]

    def prepare_cuda_graph_lora_params(
            self, scheduled_requests: "ScheduledRequests",
            attn_metadata: "AttentionMetadata",
            peft_table: "PeftTable") -> Optional[Dict]:
        """
        Prepare CUDA Graph compatible LoRA parameters from scheduled requests.

        This method replaces the original _get_lora_params_from_requests for CUDA Graph mode.

        Args:
            scheduled_requests: The scheduled requests for the current batch
            attn_metadata: Attention metadata containing batch information
            peft_table: PEFT table from cache manager

        Returns:
            Dictionary containing CUDA Graph compatible LoRA parameters, or None if no LoRA
        """
        request_list = scheduled_requests.context_requests + scheduled_requests.generation_requests

        # Check if any requests have LoRA
        has_lora_requests = any(
            hasattr(req, 'lora_task_id') and req.lora_task_id is not None
            for req in request_list)

        if not has_lora_requests:
            return None

        batch_size = len(request_list)

        # Get slot assignments for this batch
        request_to_slot_id = self.adapter_slot_manager.get_slot_mapping_for_batch(
            scheduled_requests)

        # Count tokens per slot (excluding base model requests at max_lora_size)
        slot_counts = Counter()
        for slot_id in request_to_slot_id.values():
            if slot_id != self.max_lora_size:  # Skip base model requests
                slot_counts[slot_id] += 1

        # Get or create lora_params for this batch size
        cuda_graph_lora_params = self.get_or_create_lora_params(batch_size)

        # Update slot IDs in lora_params
        slot_ids_list = [
            request_to_slot_id.get(req.py_request_id, 0) for req in request_list
        ]
        cuda_graph_lora_params.update_slot_ids(slot_ids_list, batch_size)

        # Get current slot to task mapping
        slot_to_task_mapping = self.adapter_slot_manager.get_slot_to_task_mapping(
        )

        # Update weight pointers if slot assignments changed
        if self.adapter_slot_manager.has_slots_changed():
            cuda_graph_lora_params.update_weight_pointers(
                peft_table, slot_to_task_mapping)
            self.adapter_slot_manager.reset_changed_flag()

        # Update GEMM sizes and offsets based on current batch
        # Get input hidden size from attention metadata if available
        input_hidden_size = getattr(attn_metadata, 'hidden_size',
                                    4096)  # Default fallback

        cuda_graph_lora_params.update_gemm_sizes_and_offsets(
            batch_size=batch_size,
            input_hidden_size=input_hidden_size,
            slot_counts=slot_counts,
            slot_to_task_mapping=slot_to_task_mapping,
            peft_table=peft_table)

        # Create return dictionary compatible with current LoRA layer interface
        # This bridges the old and new interfaces
        lora_params = {
            'cuda_graph_params': cuda_graph_lora_params,
            'slot_ids': cuda_graph_lora_params.slot_ids,
            'host_request_types': attn_metadata.host_request_types,
            'prompt_lens_cpu': attn_metadata.prompt_lens_cpu,
            'num_seqs': attn_metadata.num_seqs,
            'use_cuda_graph_mode': True,  # Flag to indicate new mode
        }

        return lora_params

    def prepare_legacy_lora_params(
            self, scheduled_requests: "ScheduledRequests",
            attn_metadata: "AttentionMetadata") -> Optional[Dict]:
        """
        Prepare legacy LoRA parameters for non-CUDA Graph mode.

        This method maintains the original _get_lora_params_from_requests logic
        for backward compatibility when CUDA Graph is not used.

        Args:
            scheduled_requests: The scheduled requests for the current batch
            attn_metadata: Attention metadata containing batch information

        Returns:
            Dictionary containing legacy LoRA parameters, or None if no LoRA
        """
        lora_params = {}
        tmp_lora_params = {}

        request_list = scheduled_requests.context_requests + scheduled_requests.generation_requests

        # trace all requests to get the union set of the lora params
        for request in request_list:
            if request.py_lora_task_layer_module_configs is None:
                continue

            for module in request.py_lora_task_layer_module_configs:
                module_id = module.module_id
                layer_id = module.layer_id

                if layer_id not in lora_params:
                    lora_params[layer_id] = {}
                if module_id not in lora_params[layer_id]:
                    lora_params[layer_id][module_id] = {
                        'adapter_size': [],
                        'weight_pointers': [],
                    }

                scaling_vec_pointer = module.scaling_vec_pointer
                if scaling_vec_pointer is None:
                    scaling_vec_pointer = 0
                tmp_lora_params[(request.py_request_id, layer_id,
                                 module_id)] = {
                                     'adapter_size': [module.adapter_size],
                                     'weight_pointers': [
                                         module.weights_in_pointer,
                                         module.weights_out_pointer,
                                         scaling_vec_pointer
                                     ],
                                 }

        for request in request_list:
            # Need to set default values for this case
            if request.py_lora_task_layer_module_configs is None:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        current_lora_params = lora_params[layer_id][module_id]
                        current_lora_params['adapter_size'].append(0)
                        current_lora_params['weight_pointers'] += [0, 0, 0]
            else:
                for layer_id in lora_params:
                    for module_id in lora_params[layer_id]:
                        current_tmp_lora_params = tmp_lora_params.get(
                            (request.py_request_id, layer_id, module_id), None)
                        current_lora_params = lora_params[layer_id][module_id]
                        if current_tmp_lora_params is None:
                            current_lora_params['adapter_size'].append(0)
                            current_lora_params['weight_pointers'] += [0, 0, 0]
                        else:
                            current_lora_params[
                                'adapter_size'] += current_tmp_lora_params[
                                    'adapter_size']
                            current_lora_params[
                                'weight_pointers'] += current_tmp_lora_params[
                                    'weight_pointers']

        for layer_id in lora_params:
            for module_id in lora_params[layer_id]:
                current_lora_params = lora_params[layer_id][module_id]
                current_lora_params['adapter_size'] = torch.IntTensor(
                    current_lora_params['adapter_size'])
                current_lora_params['weight_pointers'] = torch.LongTensor(
                    current_lora_params['weight_pointers'])

        if lora_params:
            lora_params['host_request_types'] = attn_metadata.host_request_types
            lora_params['prompt_lens_cpu'] = attn_metadata.prompt_lens_cpu
            lora_params['num_seqs'] = attn_metadata.num_seqs
            lora_params['use_cuda_graph_mode'] = False  # Flag for legacy mode

        return lora_params if lora_params else None
