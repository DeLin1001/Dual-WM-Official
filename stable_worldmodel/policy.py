from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable
import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors
import stable_worldmodel as swm
from stable_worldmodel.solver import Solver
from stable_worldmodel.solver.spaces import is_discrete_space
from stable_worldmodel.protocols import Actionable, Transformable

@dataclass(frozen=True)
class PlanConfig:
    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True

    @property
    def plan_len(self) -> int:
        return self.horizon * self.action_block

class BasePolicy:
    env: Any
    type: str

    def __init__(self, **kwargs: Any) -> None:
        self.env = None
        self.type = 'base'
        for (arg, value) in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        self.env = env

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        out = {}
        skip_scaled: set[str] = set()
        env = getattr(self, 'env', None)
        if getattr(self, 'process', None) and 'action' in self.process and (env is not None) and is_discrete_space(env.single_action_space):
            skip_scaled.add('action')
        for (k, v) in info_dict.items():
            is_numpy = isinstance(v, np.ndarray | np.generic)
            if hasattr(self, 'process') and k in self.process and (k not in skip_scaled):
                if not is_numpy:
                    raise ValueError(f"Expected numpy array for key '{k}' in process, got {type(v)}")
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])
                v = self.process[k].transform(v)
                v = v.reshape(shape)
            if hasattr(self, 'transform') and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith('pixels') or k.startswith('goal'):
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack([self.transform[k](tv_tensors.Image(x)) for x in v])
                is_numpy = isinstance(v, np.ndarray | np.generic)
                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])
            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)
            out[k] = v
        return out

class RandomPolicy(BasePolicy):

    def __init__(self, seed: int | None=None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        if self.env is not None:
            self.env.action_space.seed(seed)

class ZeroPolicy(BasePolicy):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = 'zero'

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)

class ExpertPolicy(BasePolicy):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = 'expert'

    def get_action(self, obs: Any, goal_obs: Any, **kwargs: Any) -> np.ndarray | None:
        pass

class FeedForwardPolicy(BasePolicy):

    def __init__(self, model: Actionable, process: dict[str, Transformable] | None=None, transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None=None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = 'feed_forward'
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"
        info_dict = self._prepare_info(info_dict)
        if 'goal' in info_dict:
            info_dict['goal_pixels'] = info_dict['goal']
        device = next(self.model.parameters()).device
        for (k, v) in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(device)
        with torch.no_grad():
            action = self.model.get_action(info_dict)
        if torch.is_tensor(action):
            action = action.cpu().detach().numpy()
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)
        return action

class WorldModelPolicy(BasePolicy):

    def __init__(self, solver: Solver, config: PlanConfig, process: dict[str, Transformable] | None=None, transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None=None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: list[deque[torch.Tensor]] | None = None
        self._next_init: torch.Tensor | None = None

    @property
    def flatten_receding_horizon(self) -> int:
        return self.cfg.receding_horizon * self.cfg.action_block

    def set_env(self, env: Any) -> None:
        self.env = env
        n_envs = getattr(env, 'num_envs', 1)
        self.solver.configure(action_space=env.action_space, n_envs=n_envs, config=self.cfg)
        normalizer = self.process.get('action') if self.process else None
        if normalizer is not None and hasattr(self.solver, 'set_action_normalizer'):
            mean = getattr(normalizer, 'mean_', None)
            scale = getattr(normalizer, 'scale_', None)
            if mean is not None and scale is not None:
                self.solver.set_action_normalizer(np.asarray(mean), np.asarray(scale))
        self._action_buffer = [deque(maxlen=self.flatten_receding_horizon) for _ in range(n_envs)]
        assert isinstance(self.solver, Solver), 'Solver must implement the Solver protocol'

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        info_dict = self._prepare_info(info_dict)
        n_envs = self.env.num_envs
        is_discrete = is_discrete_space(self.env.single_action_space)
        needs_flush = info_dict.pop('_needs_flush', None)
        if needs_flush is not None:
            for i in range(n_envs):
                if needs_flush[i]:
                    self._action_buffer[i].clear()
                    if self._next_init is not None:
                        self._next_init[i] = -1 if is_discrete else 0
        terminated = info_dict.get('terminated')
        dead = np.asarray(terminated, dtype=bool) if terminated is not None else np.zeros(n_envs, dtype=bool)
        replan_idx = [i for i in range(n_envs) if len(self._action_buffer[i]) == 0 and (not dead[i])]
        if replan_idx:
            idx_tensor = torch.as_tensor(replan_idx, dtype=torch.long)
            sliced = {}
            for (k, v) in info_dict.items():
                if torch.is_tensor(v):
                    sliced[k] = v[idx_tensor]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[replan_idx]
                elif isinstance(v, list):
                    sliced[k] = [v[i] for i in replan_idx]
                else:
                    sliced[k] = v
            sliced_init = self._next_init[idx_tensor] if self._next_init is not None else None
            sliced['_env_indices'] = idx_tensor
            if needs_flush is not None:
                sliced['_needs_flush'] = needs_flush[idx_tensor]
            outputs = self.solver(sliced, init_action=sliced_init)
            actions = outputs['actions']
            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            if self.cfg.warm_start and rest.shape[1] > 0:
                if self._next_init is None:
                    self._next_init = torch.zeros(n_envs, rest.shape[1], rest.shape[2], dtype=rest.dtype)
                self._next_init[idx_tensor] = rest
            elif not self.cfg.warm_start:
                self._next_init = None
            plan = plan.reshape(len(replan_idx), self.flatten_receding_horizon, -1)
            for (row, env_i) in enumerate(replan_idx):
                self._action_buffer[env_i].extend(plan[row])
        single_shape = self.env.single_action_space.shape
        action = torch.full((n_envs, *single_shape), fill_value=0 if is_discrete else float('nan'), dtype=torch.long if is_discrete else torch.float32)
        for i in range(n_envs):
            if not dead[i]:
                action[i] = self._action_buffer[i].popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()
        if 'action' in self.process and (not is_discrete):
            action = self.process['action'].inverse_transform(action)
        return action

def _load_model_with_attribute(run_name, attribute_name, cache_dir=None):
    if Path(run_name).exists():
        run_path = Path(run_name)
    else:
        run_path = Path(cache_dir or swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_name)
    if run_path.is_dir():
        ckpt_files = list(run_path.glob('*_object.ckpt'))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        path = ckpt_files[0]
        logging.info(f'Loading model from checkpoint: {path}')
    else:
        path = Path(f'{run_path}_object.ckpt')
        assert path.exists(), f'Checkpoint path does not exist: {path}. Launch pretraining first.'
    spt_module = torch.load(path, weights_only=False, map_location='cpu')

    def scan_module(module):
        if hasattr(module, attribute_name):
            if isinstance(module, torch.nn.Module):
                module = module.eval()
            return module
        for child in module.children():
            result = scan_module(child)
            if result is not None:
                return result
        return None
    result = scan_module(spt_module)
    if result is not None:
        return result
    raise RuntimeError(f"No module with '{attribute_name}' found in the loaded world model.")

def AutoActionableModel(run_name: str, cache_dir: str | Path | None=None) -> torch.nn.Module:
    return _load_model_with_attribute(run_name, 'get_action', cache_dir)

def AutoCostModel(run_name: str, cache_dir: str | Path | None=None) -> torch.nn.Module:
    return _load_model_with_attribute(run_name, 'get_cost', cache_dir)
Policy = BasePolicy
