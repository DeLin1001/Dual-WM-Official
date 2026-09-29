from typing import Protocol, runtime_checkable
import numpy as np
import torch

class Costable(Protocol):

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        ...

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        ...

class Transformable(Protocol):

    def transform(self, x: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        ...

@runtime_checkable
class Actionable(Protocol):

    def get_action(self, info: dict, horizon: int=1, prefix_actions: torch.Tensor | None=None) -> torch.Tensor:
        ...
