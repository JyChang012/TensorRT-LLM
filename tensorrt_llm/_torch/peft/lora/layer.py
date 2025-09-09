from enum import IntEnum
from typing import Dict, List, Optional

import torch

from ...pyexecutor.cuda_graph_lora_params import CudaGraphLoraParams


class LoraModuleType(IntEnum):
    """Enum class representing different types of modules that can have LoRA adapters.

    This enum maps to the different attention and MLP components in a transformer model
    that can be adapted using LoRA weights.
    """
    ATTENTION_QKV = 0  # Combined QKV projection
    ATTENTION_Q = 1  # Query projection
    ATTENTION_K = 2  # Key projection
    ATTENTION_V = 3  # Value projection
    ATTENTION_DENSE = 4  # Output projection after attention

    MLP_H_TO_4H = 5  # First MLP projection (hidden to 4x hidden)
    MLP_4H_TO_H = 6  # Second MLP projection (4x hidden back to hidden)
    MLP_GATE = 7  # Gate projection in MLP

    CROSS_ATTENTION_QKV = 8  # Cross-attention QKV projection
    CROSS_ATTENTION_Q = 9  # Cross-attention Query projection
    CROSS_ATTENTION_K = 10  # Cross-attention Key projection
    CROSS_ATTENTION_V = 11  # Cross-attention Value projection
    CROSS_ATTENTION_DENSE = 12  # Cross-attention output projection

    MOE_H_TO_4H = 13  # MoE first projection
    MOE_4H_TO_H = 14  # MoE second projection
    MOE_GATE = 15  # MoE gate projection
    MOE_ROUTER = 16  # MoE router

    MLP_ROUTER = 17  # MLP router
    MLP_GATE_UP = 18  # Combined gate and up projections

    def __str__(self):
        """Return the name of the enum value."""
        return self.name

    @classmethod
    def from_string(cls, name: str) -> "LoraModuleType":
        """Convert a string to the corresponding LoraModuleType.

        Args:
            name: The string name of the module type

        Returns:
            The corresponding LoraModuleType enum value

        Raises:
            ValueError: If the name doesn't match any LoraModuleType
        """
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown LoRA module type: {name}")

    @property
    def is_attention(self) -> bool:
        """Check if this is an attention module type."""
        return self in {
            self.ATTENTION_QKV, self.ATTENTION_Q, self.ATTENTION_K,
            self.ATTENTION_V, self.ATTENTION_DENSE, self.CROSS_ATTENTION_QKV,
            self.CROSS_ATTENTION_Q, self.CROSS_ATTENTION_K,
            self.CROSS_ATTENTION_V, self.CROSS_ATTENTION_DENSE
        }

    @property
    def is_mlp(self) -> bool:
        """Check if this is an MLP module type."""
        return self in {
            self.MLP_H_TO_4H, self.MLP_4H_TO_H, self.MLP_GATE, self.MLP_GATE_UP,
            self.MLP_ROUTER
        }

    @property
    def is_moe(self) -> bool:
        """Check if this is a Mixture of Experts (MoE) module type."""
        return self in {
            self.MOE_H_TO_4H, self.MOE_4H_TO_H, self.MOE_GATE, self.MOE_ROUTER
        }


class LoraLayer(torch.nn.Module):

    def __init__(self, lora_module_types: List[LoraModuleType],
                 output_hidden_sizes: List[int]):
        super().__init__()

        self.lora_module_types = lora_module_types
        self.output_hidden_sizes = output_hidden_sizes
        assert len(lora_module_types) == len(output_hidden_sizes)

    def forward(
        self,
        x,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:

        if bool(lora_params):
            # Check if we're using CUDA Graph mode
            use_cuda_graph_mode = lora_params.get('use_cuda_graph_mode', False)

            if use_cuda_graph_mode:
                return self._forward_cuda_graph_mode(x, lora_params, layer_idx)
            else:
                return self._forward_legacy_mode(x, lora_params, layer_idx)
        else:
            return None

    def _forward_cuda_graph_mode(
        self,
        x: torch.Tensor,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Forward pass using CUDA Graph compatible LoRA parameters.

        Args:
            x: Input tensor
            lora_params: CUDA Graph compatible LoRA parameters
            layer_idx: Current layer index

        Returns:
            LoRA output tensor or None
        """
        cuda_graph_params: CudaGraphLoraParams = lora_params.get(
            'cuda_graph_params')
        if cuda_graph_params is None:
            return None

        # Get layer-specific parameters
        layer_key = CudaGraphLoraParams.LoraLayerKey(
            layer_idx=layer_idx, module_ids=tuple(self.lora_module_types))
        layer_params = cuda_graph_params.get_layer_params(layer_key)

        # Skip layers that don't have LoRA modules
        if layer_params is None:
            return 0  # Pass-through for layers without LoRA modules

        # Use the new CUDA Graph compatible grouped GEMM operation
        # This will be implemented as a new custom operator
        try:
            # Create intermediate buffers sized for LoRA operations only (no base model)
            batch_size, hidden_size = x.shape[0], x.shape[-1]

            # Get max_rank from layer configuration and number of modules
            num_layer_modules = len(self.lora_module_types)
            max_rank = cuda_graph_params.max_rank

            # Intermediate buffer: [num_layer_modules, batch_size, max_rank]
            intermediate_buffer = torch.zeros(
                [num_layer_modules, batch_size, max_rank],
                dtype=x.dtype,
                device=x.device)

            # Output buffer: includes space for ALL tokens (LoRA + base model)
            # Base model tokens will be handled by pass-through in reordering kernels
            total_output_size = sum(self.output_hidden_sizes)
            output_buffer = torch.zeros(batch_size,
                                        total_output_size,
                                        dtype=x.dtype,
                                        device=x.device)

            # Call the new CUDA Graph compatible operator with sorted indices and precomputed leading dimensions
            lora_outputs = torch.ops.trtllm.lora_grouped_gemm_cuda_graph(
                x,  # Input tensor
                layer_params.d_lora_in_sizes,  # GEMM sizes for lora_in
                layer_params.d_lora_out_sizes,  # GEMM sizes for lora_out
                layer_params.d_a_offset,  # Input offsets
                layer_params.d_b_ptrs,  # Lora_in weight pointers
                layer_params.d_d_offset,  # Intermediate output offsets
                layer_params.d_b_prime_ptrs,  # Lora_out weight pointers
                layer_params.d_d_prime_offset,  # Final output offsets
                cuda_graph_params.slot_ids,  # Slot IDs (for reference)
                cuda_graph_params.
                sorted_ids[:batch_size],  # Sorted indices for gather/scatter
                cuda_graph_params.get_problem_count(
                    layer_idx),  # Number of GEMM problems
                intermediate_buffer,  # Intermediate buffer (LoRA only)
                output_buffer,  # Output buffer (all tokens)
                layer_params.d_ld_a,  # Leading dimensions for A matrices
                layer_params.d_ld_b,  # Leading dimensions for B matrices
                layer_params.
                d_ld_d,  # Leading dimensions for C matrices (reusing d_ld_d as placeholder)
                layer_params.d_ld_b_prime,
                layer_params.d_ld_d_prime,
            )
            return lora_outputs
        except AttributeError:
            # Fallback to legacy mode if new operator is not available
            return self._forward_legacy_mode(x, lora_params, layer_idx)

    def _forward_legacy_mode(
        self,
        x: torch.Tensor,
        lora_params: Dict,
        layer_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Legacy forward pass using the original LoRA implementation.

        Args:
            x: Input tensor
            lora_params: Legacy LoRA parameters
            layer_idx: Current layer index

        Returns:
            LoRA output tensor or None
        """
        lora_ranks = []
        lora_weight_pointers = []
        active_lora_module_ids = []

        for module_idx in self.lora_module_types:
            module_idx = int(module_idx)
            if module_idx in lora_params[layer_idx]:
                active_lora_module_ids.append(module_idx)
                lora_ranks.append(
                    lora_params[layer_idx][module_idx]['adapter_size'])
                lora_weight_pointers.append(
                    lora_params[layer_idx][module_idx]['weight_pointers'])

        num_seqs = lora_params['num_seqs']

        if len(active_lora_module_ids) == 0:
            return None
        else:
            lora_outputs = torch.ops.trtllm.lora_grouped_gemm(
                x,
                lora_params['host_request_types'][:num_seqs],
                lora_ranks,
                lora_weight_pointers,
                lora_params['prompt_lens_cpu'][:num_seqs],
                self.output_hidden_sizes,
                False,  # transA
                True,  # transB
                max([r.max() for r in lora_ranks]),
                0,
                True,  # TODO smor- should be lora_params["remove_input_padding"], support in loraOp as well
            )
            if isinstance(lora_outputs, torch.Tensor):
                return lora_outputs
            else:
                # For multiple LoRA modules, some might not be executed in grouped gemm.
                # For those modules not executed, we create zero tensors with matching dimensions.
                # Finally we concatenate all tensors (both LoRA outputs and zero tensors) in order.
                lora_output = []
                for module_idx in self.lora_module_types:
                    if int(module_idx) in active_lora_module_ids:
                        lora_output.append(lora_outputs.pop(0))
                    else:
                        lora_output.append(
                            torch.zeros(list(x.shape[:-1]) + [
                                self.output_hidden_sizes[
                                    self.lora_module_types.index(module_idx)]
                            ],
                                        dtype=x.dtype,
                                        device=x.device))
                lora_output = torch.cat(lora_output, dim=-1)
                return lora_output
