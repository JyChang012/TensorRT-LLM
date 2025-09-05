"""
AdapterSlotManager for CUDA Graph + multi LoRA support.

This module manages adapter slots to enable CUDA Graph compatibility with multi-LoRA
by maintaining a fixed number of slots for different LoRA adapters (task_ids).
"""

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
        self.task_id_to_slot_id: Dict[int, int] = {}  # Maps task_id -> slot_id
        self.slot_id_to_task_id: Dict[int, Optional[int]] = {
            i: None
            for i in range(max_lora_size)
        }  # Maps slot_id -> task_id

        # LRU tracking for eviction policy
        self.access_order: List[int] = [
        ]  # Most recently used task_ids, newest last

        # State tracking
        self.slots_changed = False

    def get_slot_assignment(self, batch_task_ids: Set[int]) -> Dict[int, int]:
        """
        Get slot assignments for a batch, updating slots if necessary.

        Args:
            batch_task_ids: Set of task_ids needed for the current batch

        Returns:
            Dict mapping task_id to slot_id for all task_ids in the batch
        """
        self.slots_changed = False
        missing_task_ids = batch_task_ids - set(self.task_id_to_slot_id.keys())

        if missing_task_ids:
            self._assign_slots_for_missing_tasks(missing_task_ids)

        # Update access order for all batch task_ids
        for task_id in batch_task_ids:
            if task_id in self.access_order:
                self.access_order.remove(task_id)
            self.access_order.append(task_id)

        return {
            task_id: self.task_id_to_slot_id[task_id]
            for task_id in batch_task_ids
        }

    def _assign_slots_for_missing_tasks(self, missing_task_ids: Set[int]):
        """
        Assign slots for missing task_ids, evicting if necessary.

        Args:
            missing_task_ids: Set of task_ids that need slot assignments
        """
        self.slots_changed = True

        for task_id in missing_task_ids:
            # Find an available slot or evict LRU
            available_slot = self._find_available_slot()
            if available_slot is not None:
                # Use available slot
                self._assign_slot(task_id, available_slot)
            else:
                # Evict LRU task_id
                lru_task_id = self._get_lru_task_id()
                if lru_task_id is not None:
                    lru_slot_id = self.task_id_to_slot_id[lru_task_id]
                    self._evict_slot(lru_slot_id)
                    self._assign_slot(task_id, lru_slot_id)

    def _find_available_slot(self) -> Optional[int]:
        """Find an available (empty) slot."""
        for slot_id, task_id in self.slot_id_to_task_id.items():
            if task_id is None:
                return slot_id
        return None

    def _get_lru_task_id(self) -> Optional[int]:
        """Get the least recently used task_id that's currently in a slot."""
        for task_id in self.access_order:
            if task_id in self.task_id_to_slot_id:
                return task_id
        return None

    def _assign_slot(self, task_id: int, slot_id: int):
        """Assign a task_id to a slot_id."""
        self.task_id_to_slot_id[task_id] = slot_id
        self.slot_id_to_task_id[slot_id] = task_id

    def _evict_slot(self, slot_id: int):
        """Evict the task_id currently in the given slot."""
        task_id = self.slot_id_to_task_id[slot_id]
        if task_id is not None:
            del self.task_id_to_slot_id[task_id]
            self.slot_id_to_task_id[slot_id] = None
            if task_id in self.access_order:
                self.access_order.remove(task_id)

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

        # Get slot assignments for non-base-model task_ids
        if batch_task_ids:
            task_id_to_slot_id = self.get_slot_assignment(batch_task_ids)
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

    def get_active_slots(self) -> List[int]:
        """
        Get list of currently active (occupied) slot IDs.

        Returns:
            List of slot_ids that are currently occupied by task_ids
        """
        return [
            slot_id for slot_id, task_id in self.slot_id_to_task_id.items()
            if task_id is not None
        ]

    def get_slot_to_task_mapping(self) -> Dict[int, Optional[int]]:
        """
        Get current slot to task mapping.

        Returns:
            Dict mapping slot_id to task_id (or None if slot is empty)
        """
        return self.slot_id_to_task_id.copy()

    def has_slots_changed(self) -> bool:
        """Check if slot assignments have changed since last check."""
        return self.slots_changed

    def reset_changed_flag(self):
        """Reset the slots_changed flag."""
        self.slots_changed = False
