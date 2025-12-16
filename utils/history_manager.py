"""
History manager for undo/redo functionality.
"""

import numpy as np
from typing import Optional, List, Tuple


class HistoryManager:
    """
    Manages undo/redo for image editing operations.

    Stores image states and operation descriptions.
    """

    def __init__(self, max_history: int = 20):
        """
        Initialize history manager.

        Args:
            max_history: Maximum number of states to keep in history
        """
        self.history: List[Tuple[np.ndarray, str]] = []
        self.current_index = -1
        self.max_history = max_history

    def add_state(self, image: np.ndarray, description: str):
        """
        Add a new state to history.

        Args:
            image: Image array to save
            description: Description of the operation
        """
        # Remove any states after current_index (when undoing then making new changes)
        self.history = self.history[:self.current_index + 1]

        # Add new state
        self.history.append((image.copy(), description))

        # Limit history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.current_index += 1

    def undo(self) -> Optional[np.ndarray]:
        """
        Go back one state.

        Returns:
            Previous image state, or None if can't undo
        """
        if self.can_undo():
            self.current_index -= 1
            return self.history[self.current_index][0].copy()
        return None

    def redo(self) -> Optional[np.ndarray]:
        """
        Go forward one state.

        Returns:
            Next image state, or None if can't redo
        """
        if self.can_redo():
            self.current_index += 1
            return self.history[self.current_index][0].copy()
        return None

    def can_undo(self) -> bool:
        """Check if undo is possible."""
        return self.current_index > 0

    def can_redo(self) -> bool:
        """Check if redo is possible."""
        return self.current_index < len(self.history) - 1

    def get_current(self) -> Optional[np.ndarray]:
        """
        Get current image state.

        Returns:
            Current image, or None if history is empty
        """
        if self.history and 0 <= self.current_index < len(self.history):
            return self.history[self.current_index][0].copy()
        return None

    def get_history_list(self) -> List[str]:
        """
        Get list of operation descriptions up to current state.

        Returns:
            List of operation descriptions
        """
        return [desc for _, desc in self.history[:self.current_index + 1]]

    def get_full_history_list(self) -> List[str]:
        """
        Get complete list of operation descriptions.

        Returns:
            List of all operation descriptions
        """
        return [desc for _, desc in self.history]

    def jump_to_state(self, index: int) -> Optional[np.ndarray]:
        """
        Jump to a specific state in history.

        Args:
            index: History index to jump to

        Returns:
            Image at specified index, or None if invalid
        """
        if 0 <= index < len(self.history):
            self.current_index = index
            return self.history[self.current_index][0].copy()
        return None

    def reset(self):
        """Clear all history."""
        self.history = []
        self.current_index = -1

    def get_current_index(self) -> int:
        """Get current history index."""
        return self.current_index

    def get_history_length(self) -> int:
        """Get total number of states in history."""
        return len(self.history)
