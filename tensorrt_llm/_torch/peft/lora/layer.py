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
    PTR_DTYPE = torch.int64
    LD_DTYPE = torch.int64
    SIZES_DTYPE = torch.int32

    def __init__(self, lora_module_types: List[LoraModuleType],
                 output_hidden_sizes: List[int]):
        super().__init__()

        self.lora_module_types = lora_module_types
        self.output_hidden_sizes = torch.tensor(output_hidden_sizes,
                                                dtype=self.SIZES_DTYPE)
        self.output_hidden_sizes_list = output_hidden_sizes
        assert len(lora_module_types) == len(output_hidden_sizes)
        self.output_sizes_offset = CudaGraphLoraParams.get_offset_from_counts(
            self.output_hidden_sizes).to(
                dtype=self.PTR_DTYPE)  # [num_layer_modules]
        self.output_sizes_offset_device = self.output_sizes_offset.to(
            device='cuda')
        self.output_hidden_size_device = self.output_hidden_sizes.to(
            device='cuda')

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

    def prepare_grouped_gemm_buffers(
            self, x: torch.Tensor, cuda_graph_lora_params: CudaGraphLoraParams,
            layer_key: int):
        device = x.device
        bs, input_hidden_size = x.shape
        assert input_hidden_size == cuda_graph_lora_params.layer_info[
            layer_key].input_hidden_size
        shape_2d = (len(self.lora_module_types),
                    cuda_graph_lora_params.max_lora_size
                    )  # [num_layer_modules, max_lora_size]
        shape_3d = shape_2d + (3, )
        sum_out_sizes = sum(self.output_hidden_sizes)

        # a [bs, hidden]
        lda = torch.full(shape_2d,
                         input_hidden_size,
                         dtype=self.LD_DTYPE,
                         device=device)

        # b [input_hidden_size, lora_rank]
        ldb = lda

        # a_prime / d [num_layer_modules, bs, max_rank]
        ldd = torch.full(shape_2d,
                         cuda_graph_lora_params.max_rank,
                         dtype=self.LD_DTYPE,
                         device=device)

        # b_prime [lora_rank, module_output_size]
        ldb_prime = cuda_graph_lora_params.slot_ranks.unsqueeze(0).repeat(
            shape_2d[0], 1)

        # d_prime [bs, sum_of_each_module_output_sizes]
        ldd_prime = torch.full(shape_2d,
                               sum_out_sizes,
                               dtype=self.LD_DTYPE,
                               device=device)

        # reordered a [bs, hidden], each module has the same offset
        a_offset = cuda_graph_lora_params.slot_offsets * input_hidden_size
        a_offset = a_offset.unsqueeze(0).repeat(shape_2d[0], 1)

        # d [num_layer_modules, bs, max_rank]
        d_offset = (
            cuda_graph_lora_params.slot_offsets.unsqueeze(0) + torch.arange(
                shape_2d[0], device=device, dtype=self.PTR_DTYPE).unsqueeze(1) *
            bs) * cuda_graph_lora_params.max_rank

        # d' [bs, sum_of_each_module_output_sizes]
        bs_offset = cuda_graph_lora_params.slot_offsets.unsqueeze(
            0)  # [1, max_lora_size]
        bs_offset = bs_offset * sum_out_sizes
        out_offset = self.output_sizes_offset_device.unsqueeze(
            1)  # [num_layer_modules, 1]
        d_prime_offset = bs_offset + out_offset

        # sizes
        in_sizes = torch.empty(shape_3d, dtype=self.SIZES_DTYPE, device=device)
        out_sizes = torch.empty_like(in_sizes)

        slot_counts = cuda_graph_lora_params.slot_counts.unsqueeze(
            0)  # [1, max_lora_size]
        ranks = cuda_graph_lora_params.slot_ranks.unsqueeze(
            0)  # [1, max_lora_size]
        output_hidden_sizes = self.output_hidden_size_device.unsqueeze(
            1)  # [num_layer_modules, 1]

        in_sizes[:, :, 0] = slot_counts
        in_sizes[:, :, 1] = ranks
        in_sizes[:, :, 2].fill_(input_hidden_size)

        out_sizes[:, :, 0] = slot_counts
        out_sizes[:, :, 1] = output_hidden_sizes
        out_sizes[:, :, 2] = ranks

        # splitk offtsets (m * n) for the first grouped gemm with (m, n, k) = (slot_counts, slot_ranks, input_hidden_size)
        splitk_offsets = torch.zeros(cuda_graph_lora_params.max_lora_size,
                                     dtype=self.LD_DTYPE,
                                     device=device)
        splitk_offsets[1:] = cuda_graph_lora_params.slot_counts[:-1]
        splitk_offsets[1:] *= cuda_graph_lora_params.slot_ranks[:-1]
        splitk_offsets[1:].cumsum_(dim=0)

        # dummy max sizes
        host_max_in_sizes = torch.empty(
            shape_3d, dtype=self.SIZES_DTYPE
        )  # m: batch_size, n: max_lora_rank, k: input_hidden_size
        host_max_out_sizes = torch.empty_like(
            host_max_in_sizes
        )  # m: batch_size, n: max_output_hidden_size, k: max_lora_rank
        host_max_in_sizes[:, :, 0] = bs
        host_max_in_sizes[:, :, 1] = cuda_graph_lora_params.max_rank
        host_max_in_sizes[:, :, 2] = input_hidden_size

        host_max_out_sizes[:, :, 0] = bs
        host_max_out_sizes[:, :, 1] = self.output_hidden_sizes.unsqueeze(1)
        host_max_out_sizes[:, :, 2] = cuda_graph_lora_params.max_rank

        return in_sizes, out_sizes, a_offset, d_offset, d_prime_offset, lda, ldb, ldd, ldb_prime, ldd_prime, host_max_in_sizes, host_max_out_sizes, splitk_offsets  # TODO: compute ptr of offsets directly

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
        # Get layer-specific parameters
        layer_key = CudaGraphLoraParams.LoraLayerKey(
            layer_idx=layer_idx, module_ids=tuple(self.lora_module_types))

        if cuda_graph_params is None or not cuda_graph_params.layer_info[
                layer_key].is_enabled():
            return None

        layer_params = cuda_graph_params.get_layer_params(layer_key)

        # Skip layers that don't have LoRA modules
        if layer_params is None:
            return 0  # Pass-through for layers without LoRA modules

        # Use the new CUDA Graph compatible grouped GEMM operation
        # This will be implemented as a new custom operator
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

        in_sizes, out_sizes, a_offset, d_offset, d_prime_offset, lda, ldb, ldd, ldb_prime, ldd_prime, host_max_in_sizes, host_max_out_sizes, splitk_offsets = self.prepare_grouped_gemm_buffers(
            x, cuda_graph_params, layer_key)
        '''
        # check sizes are in range
        # in_sizes: [num_layer_modules, max_lora_size, (slot_counts, ranks, input_hidden_size)]
        # out_sizes: [num_layer_modules, max_lora_size, (slot_counts, output_hidden_sizes, ranks)]
        a_mem = in_sizes[:, :, 0] * lda
        d_mem = in_sizes[:, :, 0] * ldd

        a_max_mem = a_mem + a_offset
        a_max_mem = a_max_mem.max().cpu()

        d_max_mem = d_mem + d_offset
        d_max_mem = d_max_mem.max().cpu()

        d_prime_max_mem = d_prime_offset + (out_sizes[:, :, 0] - 1) * ldd_prime + out_sizes[:, :, 1]
        d_prime_max_mem = d_prime_max_mem.max().cpu()

        ret1 = torch.all(a_max_mem > x.nelement()).item()
        wrong_a = torch.gather(x, 0, cuda_graph_params.sorted_ids[:batch_size].unsqueeze(1).expand([-1, x.shape[1]]))
        retx = torch.all(a_max_mem > wrong_a.nelement()).item()
        ret2 = torch.all(d_max_mem > intermediate_buffer.nelement()).item()
        ret3 = torch.all(d_prime_max_mem > output_buffer.nelement()).item()

        msg = []
        if ret1:
            msg.append(f"a out of range: max accessed mem: {a_max_mem.item()}, total mem: {x.nelement()}")
        if retx:
            msg.append(f"wrong a out of range: max accessed mem: {a_max_mem.item()}, total mem: {wrong_a.nelement()}; wrong_a.shape: {wrong_a.shape}; sorted_ids: {cuda_graph_params.sorted_ids.cpu()}; batch_size: {batch_size}")
        if ret2:
            msg.append(f"d out of range: max accessed mem: {d_max_mem.item()}, total mem: {intermediate_buffer.nelement()}")
        if ret3:
            msg.append(f"d_prime out of range: max accessed mem: {d_prime_max_mem.item()}, total mem: {output_buffer.nelement()}")
        if torch.any(layer_params.d_b_ptrs == 0):
            msg.append(f"d_b_ptrs has zeros, {layer_params.d_b_ptrs.cpu()}")
        if torch.any(layer_params.d_b_prime_ptrs == 0):
            msg.append(f"d_b_prime_ptrs has zeros, {layer_params.d_b_prime_ptrs.cpu()}")

        if msg:
            msg.append(f"in_sizes: {in_sizes.cpu()}\nout_sizes: {out_sizes.cpu()}\nlda: {lda.cpu()}\nldb: {ldb.cpu()}\nldd: {ldd.cpu()}\nldb_prime: {ldb_prime.cpu()}\nldd_prime: {ldd_prime.cpu()}")
            msg.append(f"slot_offsets: {cuda_graph_params.slot_offsets.cpu()}\nslot_counts: {cuda_graph_params.slot_counts.cpu()}\nslot_ranks: {cuda_graph_params.slot_ranks.cpu()}\noutput_hidden_sizes: {self.output_hidden_size_device.cpu()}")
            msg.append(f"d_offset: {d_offset.cpu()}\nd_prime_offset: {d_prime_offset.cpu()}")
            msg.append(f"output_sizes_offset: {self.output_sizes_offset_device.cpu()}")
        if msg:
            msg = ['=' * 100, f'LeyerKey: {layer_key}'] + msg
            msg.append("=" * 100)
            print("\n".join(msg))
            '''

        # Call the new CUDA Graph compatible operator with sorted indices and precomputed leading dimensions
        lora_outputs = torch.ops.trtllm.lora_grouped_gemm_cuda_graph(
            x,  # Input tensor
            in_sizes,  # GEMM sizes for lora_in
            out_sizes,  # GEMM sizes for lora_out
            a_offset,  # Input offsets
            layer_params.d_b_ptrs,  # Lora_in weight pointers
            d_offset,  # Intermediate output offsets
            layer_params.d_b_prime_ptrs,  # Lora_out weight pointers
            d_prime_offset,  # Final output offsets
            cuda_graph_params.slot_ids,  # Slot IDs (for reference)
            cuda_graph_params.
            sorted_ids[:batch_size],  # Sorted indices for gather/scatter
            cuda_graph_params.get_problem_count(
                layer_key),  # Number of GEMM problems
            intermediate_buffer,  # Intermediate buffer (LoRA only)
            output_buffer,  # Output buffer (all tokens)
            lda,  # Leading dimensions for A matrices
            ldb,  # Leading dimensions for B matrices
            ldd,  # Leading dimensions for C matrices (reusing d_ld_d as placeholder)
            ldb_prime,
            ldd_prime,
            host_max_in_sizes,
            host_max_out_sizes,
            splitk_offsets,
        )
        return lora_outputs

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
                self.output_hidden_sizes_list,
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
                                self.output_hidden_sizes_list[
                                    self.lora_module_types.index(module_idx)]
                            ],
                                        dtype=x.dtype,
                                        device=x.device))
                lora_output = torch.cat(lora_output, dim=-1)
                return lora_output
