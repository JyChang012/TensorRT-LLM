"""
AdapterSlotManager for CUDA Graph + multi LoRA support.

This module manages adapter slots to enable CUDA Graph compatibility with multi-LoRA
by maintaining a fixed number of slots for different LoRA adapters (task_ids).
"""

from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    pass

    from .scheduler import ScheduledRequests


class AdapterSlotManager:
    """
    Manages max_lora_sizes ordered slots for distinct task_ids to enable CUDA Graph compatibility.

    Each slot can hold one adapter (task_id) and maintains a consistent ordering that allows
    the CUDA Graph to be captured with fixed buffer layouts.
    """

    def __init__(self, max_lora_size: int, device: str = "cuda"):
        """
        Initialize the AdapterSlotManager.

        Args:
            max_lora_size: Maximum number of LoRA adapters that can be active simultaneously
            device: Device to allocate tensors on
        """
        self.max_lora_size = max_lora_size
        self.device = device

        # Slot management
        self.slot2task: List[Optional[int]] = [None] * max_lora_size
        self.task2slot: OrderedDict[int,
                                    int] = OrderedDict()  # represent LRU order

        # State tracking
        self.slots_changed = False

    def get_or_assign_slots(self, batch_task_ids: Set[int]) -> Dict[int, int]:
        """
        Get slot assignments for a batch, updating slots if necessary.

        Args:
            batch_task_ids: Set of task_ids needed for the current batch

        Returns:
            Dict mapping task_id to slot_id for all task_ids in the batch
        """
        ret = dict()
        for task_id in batch_task_ids:
            slot, _ = self.get_or_assign_slot(task_id)
            ret[task_id] = slot
        return ret

    def get_or_assign_slot(self, task_id: int) -> tuple[int, Optional[int]]:
        """
        Assign a task_id to a slot and do LRU eviction if necessary.
        If already in any slot, update LRU order.
        Return: pair (assigned slot_id, evicted task_id)
        """
        evicted_task = None
        if task_id in self.task2slot:
            self.task2slot.move_to_end(task_id)
        else:
            self.slots_changed = True
            if len(self.task2slot) < self.max_lora_size:
                self.slot2task[len(self.task2slot)] = task_id
                self.task2slot[task_id] = len(self.task2slot)
            else:
                # evict lru
                evicted_task, evicted_slot = self.task2slot.popitem(last=False)
                self.slot2task[evicted_slot] = task_id
                self.task2slot[task_id] = evicted_slot
        return self.task2slot[task_id], evicted_task

    def get_slot_mapping_for_batch(
            self, scheduled_requests: "ScheduledRequests") -> Dict[int, int]:
        """
        Get slot mapping for all requests in a scheduled batch.

        Args:
            scheduled_requests: The scheduled requests for the current batch

        Returns:
            Dict mapping request_id to slot_id, with slot_id=max_lora_size for base model requests
        """
        # Collect all task_ids from requests
        batch_task_ids = set()
        request_to_task_id = {}

        all_requests = scheduled_requests.context_requests + scheduled_requests.generation_requests

        for request in all_requests:
            if hasattr(request,
                       'lora_task_id') and request.lora_task_id is not None:
                task_id = int(request.lora_task_id)
                batch_task_ids.add(task_id)
                request_to_task_id[request.py_request_id] = task_id
            else:
                # Base model request - use special marker (no task_id)
                request_to_task_id[request.py_request_id] = None
        assert len(
            batch_task_ids
        ) <= self.max_lora_size, "Currently do not support batch with more LoRA task ID than loRA slot size!"

        # Get slot assignments for non-base-model task_ids
        if batch_task_ids:
            task_id_to_slot_id = self.get_or_assign_slots(batch_task_ids)
        else:
            task_id_to_slot_id = {}

        # Map request_ids to slot_ids
        request_to_slot_id = {}
        for request_id, task_id in request_to_task_id.items():
            if task_id is None:
                # Base model request - use slot_id = max_lora_size (dummy slot)
                request_to_slot_id[request_id] = self.max_lora_size
            else:
                # LoRA request
                request_to_slot_id[request_id] = task_id_to_slot_id[task_id]

        return request_to_slot_id

    def get_slot_to_task_mapping(self) -> Dict[int, Optional[int]]:
        """
        Get current slot to task mapping.

        Returns:
            Dict mapping slot_id to task_id (or None if slot is empty)
        """
        return {sid: tid for sid, tid in enumerate(self.slot2task)}

    def has_slots_changed(self) -> bool:
        """Check if slot assignments have changed since last check."""
        return self.slots_changed

    def reset_changed_flag(self):
        """Reset the slots_changed flag."""
        self.slots_changed = False
