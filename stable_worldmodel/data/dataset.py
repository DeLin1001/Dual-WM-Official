from __future__ import annotations
from collections.abc import Callable
from typing import Any
import numpy as np
import torch

class Dataset:

    def __init__(self, lengths: np.ndarray, offsets: np.ndarray, frameskip: int=1, num_steps: int=1, transform: Callable[[dict], dict] | None=None) -> None:
        self.lengths = lengths
        self.offsets = offsets
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip
        self.transform = transform
        self.clip_indices = [(ep, start) for ep, length in enumerate(lengths) if length >= self.span for start in range(length - self.span + 1)]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)
        if 'action' in steps:
            steps['action'] = steps['action'].reshape(self.num_steps, -1)
        return steps

    def load_chunk(self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray) -> list[dict]:
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(ep, s, e)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape((e - s) // self.frameskip, -1)
            chunk.append(steps)
        return chunk

    def load_episode(self, episode_idx: int) -> dict:
        return self._load_slice(episode_idx, 0, self.lengths[episode_idx])

    def get_col_data(self, col: str) -> np.ndarray:
        raise NotImplementedError

    def get_dim(self, col: str) -> int:
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        raise NotImplementedError

    def merge_col(self, source: list[str] | str, target: str, dim: int=-1) -> None:
        raise NotImplementedError
