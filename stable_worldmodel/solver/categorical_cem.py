import time
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from loguru import logger as logging
from stable_worldmodel.solver.callbacks import Callback
from stable_worldmodel.solver.solver import Costable

class CategoricalCEMSolver:

    def __init__(self, model: Costable, batch_size: int=1, num_samples: int=300, n_steps: int=30, topk: int=30, smoothing: float=0.0, alpha: float=0.0, warm_start_escape: float | None=None, device: str | torch.device='cpu', seed: int=1234, callbacks: list[Callback] | None=None) -> None:
        self.model = model
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.smoothing = smoothing
        self.alpha = alpha
        self.warm_start_escape = max(float(smoothing), 0.05) if warm_start_escape is None else float(warm_start_escape)
        if not np.isfinite(self.warm_start_escape) or self.warm_start_escape < 0:
            raise ValueError('warm_start_escape must be finite and non-negative')
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.callbacks = list(callbacks) if callbacks else []
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        if isinstance(action_space, gym.spaces.Discrete):
            base_simplex_dim = int(action_space.n)
        elif isinstance(action_space, gym.spaces.MultiDiscrete):
            nvec = np.asarray(action_space.nvec, dtype=np.int64).reshape(-1)
            is_batched_scalar = nvec.size == n_envs and nvec.size > 0 and bool(np.all(nvec == nvec[0]))
            if not is_batched_scalar:
                raise NotImplementedError(f'CategoricalCEMSolver currently supports scalar Discrete actions only. A genuine MultiDiscrete action requires component-wise categorical distributions; got nvec shape={np.asarray(action_space.nvec).shape}, n_envs={n_envs}.')
            base_simplex_dim = int(nvec[0])
        elif isinstance(action_space, gym.spaces.Box) and np.issubdtype(action_space.dtype, np.integer):
            raise NotImplementedError('Integer Box usually represents a batched MultiDiscrete action. Component-wise categorical planning is not yet implemented; refusing to flatten it into an incorrect scalar action distribution.')
        else:
            raise ValueError(f'Action space must be scalar Discrete (or its homogeneous batched MultiDiscrete form), got {type(action_space).__name__}')
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._base_simplex_dim = base_simplex_dim
        expected_width = getattr(getattr(self.model, 'action_encoder', None), 'input_dim', None)
        actual_width = self._base_simplex_dim * int(config.action_block)
        if expected_width is not None and int(expected_width) != actual_width:
            raise ValueError(f'Discrete planner/model action width mismatch: {self._base_simplex_dim} categories * action_block {config.action_block} = {actual_width}, but model.action_encoder expects {expected_width} features.')
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._base_simplex_dim

    @property
    def action_block(self) -> int:
        return self._config.action_block

    @property
    def base_simplex_dim(self) -> int:
        return self._base_simplex_dim

    @property
    def action_simplex_dim(self) -> int:
        return self._base_simplex_dim * self.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def init_probs(self, n_envs: int) -> torch.Tensor:
        K = self._base_simplex_dim
        return torch.full((n_envs, self.horizon, self.action_block, K), 1.0 / K, dtype=self.dtype, device=self.device)

    def warm_start_probs(self, init_action: Any, probs: torch.Tensor) -> torch.Tensor:
        if init_action is None:
            return probs
        if torch.is_tensor(init_action):
            init = init_action.detach().to(device='cpu', dtype=torch.float64)
        else:
            init = torch.as_tensor(np.asarray(init_action), dtype=torch.float64)
        if init.ndim == 1:
            init = init.unsqueeze(0)
        if init.ndim != 3 or init.shape[0] != probs.shape[0]:
            logging.debug(f'[categorical_cem] ignoring warm start with shape {tuple(init.shape)}')
            return probs
        valid: torch.Tensor
        if init.shape[2] == self._base_simplex_dim and self.action_block == 1:
            valid = torch.isfinite(init).all(dim=-1, keepdim=True) & (init.sum(dim=-1, keepdim=True) > 0)
            init = init.argmax(dim=-1, keepdim=True).to(torch.float64)
        elif init.shape[2] != self.action_block:
            logging.debug(f'[categorical_cem] ignoring warm start: {init.shape[2]} plan slots per step, expected {self.action_block}')
            return probs
        else:
            rounded = init.round()
            valid = torch.isfinite(init) & (rounded >= 0) & (rounded < self._base_simplex_dim)
        steps = min(init.shape[1], self.horizon)
        if steps == 0:
            return probs
        indices = init[:, :steps].round()
        indices = torch.where(valid[:, :steps], indices, torch.zeros_like(indices))
        indices = indices.clamp(0, self._base_simplex_dim - 1).long()
        seeded = torch.nn.functional.one_hot(indices, num_classes=self._base_simplex_dim).to(torch.float64)
        escape = self.warm_start_escape
        seeded = seeded + escape
        seeded = seeded / seeded.sum(dim=-1, keepdim=True)
        out = probs.clone()
        seeded = seeded.to(device=probs.device, dtype=probs.dtype)
        valid_seed = valid[:, :steps].to(device=probs.device).unsqueeze(-1)
        out[:, :steps] = torch.where(valid_seed, seeded, out[:, :steps])
        return out

    def _sample_indices(self, probs: torch.Tensor) -> torch.Tensor:
        (bs, H, ab, K) = probs.shape
        log_probs = probs.clamp_min(1e-10).log()
        log_probs = log_probs.unsqueeze(1).expand(bs, self.num_samples, H, ab, K)
        u = torch.rand(log_probs.shape, generator=self.torch_gen, device=self.device, dtype=self.dtype).clamp_min(1e-10)
        gumbel = -(-u.log()).log()
        return (log_probs + gumbel).argmax(dim=-1)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: Any=None) -> dict:
        start_time = time.time()
        outputs: dict = {'costs': [], 'probs': []}
        total_envs = len(next(iter(info_dict.values())))
        probs = self.init_probs(total_envs)
        probs = self.warm_start_probs(init_action, probs)
        for cb in self.callbacks:
            cb.reset()
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_probs = probs[start_idx:end_idx]
            expanded_infos: dict = {}
            for (k, v) in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    target_dtype = self.dtype if v_batch.is_floating_point() else None
                    v_batch = v_batch.to(device=self.device, dtype=target_dtype).unsqueeze(1).expand(current_bs, self.num_samples, *v_batch.shape[1:])
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = v_batch
            for cb in self.callbacks:
                cb.start_batch()
            final_batch_cost = None
            for step in range(self.n_steps):
                indices = self._sample_indices(batch_probs)
                indices[:, 0] = batch_probs.argmax(dim=-1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=self._base_simplex_dim).to(self.dtype)
                candidates = one_hot.reshape(current_bs, self.num_samples, self.horizon, self.action_simplex_dim)
                costs = self.model.get_cost(expanded_infos, candidates)
                assert isinstance(costs, torch.Tensor), f'Expected cost to be a torch.Tensor, got {type(costs)}'
                assert costs.ndim == 2 and costs.shape[0] == current_bs and (costs.shape[1] == self.num_samples), f'Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
                (topk_vals, topk_inds) = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_one_hot = one_hot[batch_indices, topk_inds]
                new_probs = topk_one_hot.mean(dim=1)
                if self.smoothing > 0:
                    new_probs = new_probs + self.smoothing
                    new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
                prev_probs = batch_probs
                if self.alpha > 0:
                    batch_probs = self.alpha * batch_probs + (1 - self.alpha) * new_probs
                else:
                    batch_probs = new_probs
                for cb in self.callbacks:
                    cb(step=step, candidates=candidates, costs=costs, topk_vals=topk_vals, topk_inds=topk_inds, topk_candidates=topk_one_hot, probs=batch_probs, prev_probs=prev_probs)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()
            probs[start_idx:end_idx] = batch_probs
            outputs['costs'].extend(final_batch_cost)
        actions = probs.argmax(dim=-1)
        outputs['actions'] = actions.detach().cpu()
        outputs['probs'] = [probs.detach().cpu()]
        outputs['solve_time'] = time.time() - start_time
        if self.callbacks:
            outputs['callbacks'] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs['callbacks'][cb.output_key] = cb.history
        return outputs
