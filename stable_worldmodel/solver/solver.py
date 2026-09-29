from typing import Any, Protocol, runtime_checkable
import gymnasium as gym
import torch
from stable_worldmodel.protocols import Costable

@runtime_checkable
class Solver(Protocol):

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        ...

    @property
    def action_dim(self) -> int:
        ...

    @property
    def n_envs(self) -> int:
        ...

    @property
    def horizon(self) -> int:
        ...

    def solve(self, info_dict: dict, init_action: torch.Tensor | None=None) -> dict:
        ...
