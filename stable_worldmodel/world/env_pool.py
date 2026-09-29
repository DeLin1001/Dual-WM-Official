from __future__ import annotations
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

class EnvPool:

    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self._single_env = self.envs[0]
        self._stacked_infos: dict[str, Any] | None = None
        self.seeds = np.zeros(len(self.envs), dtype=np.int64)
        self._action_space = batch_space(self._single_env.action_space, len(self.envs))
        self._observation_space = batch_space(self._single_env.observation_space, len(self.envs))

    @property
    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    @property
    def single_action_space(self) -> gym.Space:
        return self._single_env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def single_observation_space(self) -> gym.Space:
        return self._single_env.observation_space

    @property
    def variation_space(self):
        return getattr(self._single_env.unwrapped, 'variation_space', None)

    @property
    def single_variation_space(self):
        return self.variation_space

    def reset(self, seed: int | list[int | None] | None=None, options: dict | list[dict | None] | None=None, mask: np.ndarray | None=None) -> tuple[None, dict]:
        seeds = _broadcast_arg(seed, self.num_envs, increment=True)
        opts = _broadcast_arg(options, self.num_envs)
        per_env_infos = [None] * self.num_envs
        for (i, env) in enumerate(self.envs):
            if mask is not None and (not mask[i]):
                continue
            (_, per_env_infos[i]) = env.reset(seed=seeds[i], options=opts[i])
            if seeds[i] is not None:
                self.seeds[i] = seeds[i]
        if self._stacked_infos is None or mask is None:
            self._stacked_infos = _stack_fresh(per_env_infos)
        else:
            for (i, info) in enumerate(per_env_infos):
                if info is not None:
                    _write_env_info(self._stacked_infos, i, info)
        return (None, self._stacked_infos)

    def step(self, actions: np.ndarray, mask: np.ndarray | None=None) -> tuple[None, np.ndarray, np.ndarray, np.ndarray, dict]:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        for (i, env) in enumerate(self.envs):
            if mask is not None and (not mask[i]):
                continue
            (_, rewards[i], terminateds[i], truncateds[i], info) = env.step(actions[i])
            _write_env_info(self._stacked_infos, i, info)
        return (None, rewards, terminateds, truncateds, self._stacked_infos)

    def close(self):
        for env in self.envs:
            env.close()

def _broadcast_arg(arg, n: int, increment: bool=False) -> list:

    def normalize(value):
        return int(value) if isinstance(value, np.integer) else value
    if arg is None:
        return [None] * n
    if isinstance(arg, list):
        return arg
    if isinstance(arg, np.ndarray):
        return [normalize(value) for value in arg]
    if increment and isinstance(arg, (int, np.integer)):
        return [int(arg) + i for i in range(n)]
    return [normalize(arg)] * n

def _stack_fresh(per_env_infos: list[dict]) -> dict[str, Any]:
    keys = per_env_infos[0].keys()
    stacked = {}
    for k in keys:
        vals = [info[k] for info in per_env_infos]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            stacked[k] = torch.stack(vals).unsqueeze(1)
        elif isinstance(first, np.ndarray):
            stacked[k] = np.stack(vals)[:, None, ...]
        elif isinstance(first, (bool, int, float, np.number)):
            stacked[k] = np.array(vals)[:, None]
        else:
            stacked[k] = [[v] for v in vals]
    return stacked

def _write_env_info(stacked: dict, idx: int, info: dict) -> None:
    for (k, v) in info.items():
        if k not in stacked:
            continue
        buf = stacked[k]
        if isinstance(buf, torch.Tensor):
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, dtype=buf.dtype, device=buf.device)
            buf[idx, 0] = v
        elif isinstance(buf, np.ndarray):
            buf[idx, 0] = v
        elif isinstance(buf, list):
            buf[idx][0] = v
