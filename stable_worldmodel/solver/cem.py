import time
from collections.abc import Callable
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging
from stable_worldmodel.solver.utils import prepare_init_action
from .callbacks import Callback
from .solver import Costable

def standard_normal_kl(mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if mean.shape != scale.shape:
        raise ValueError(f'mean and scale must have identical shapes, got {tuple(mean.shape)} and {tuple(scale.shape)}')
    if mean.ndim < 2:
        raise ValueError('mean and scale must include a batch dimension')
    if torch.any(scale < 0):
        raise ValueError('Gaussian scale must be non-negative')
    scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
    elementwise = 0.5 * (mean.square() + scale.square() - 1 - 2 * scale.log())
    return elementwise.flatten(start_dim=1).mean(dim=1)

def project_standard_normal_kl(mean: torch.Tensor, scale: torch.Tensor, max_mean_kl: float, bisection_steps: int=32) -> tuple[torch.Tensor, torch.Tensor]:
    if max_mean_kl < 0:
        raise ValueError(f'max_mean_kl must be non-negative, got {max_mean_kl}')
    scale = scale.clamp_min(torch.finfo(scale.dtype).eps)
    original_kl = standard_normal_kl(mean, scale)
    needs_projection = original_kl > max_mean_kl
    if not torch.any(needs_projection):
        return (mean, scale)
    log_scale = scale.log()
    lower = torch.zeros_like(original_kl)
    upper = torch.ones_like(original_kl)
    for _ in range(bisection_steps):
        alpha = (lower + upper) / 2
        view_shape = (alpha.shape[0],) + (1,) * (mean.ndim - 1)
        alpha_view = alpha.view(view_shape)
        candidate_mean = alpha_view * mean
        candidate_scale = torch.exp(alpha_view * log_scale)
        feasible = standard_normal_kl(candidate_mean, candidate_scale) <= max_mean_kl
        lower = torch.where(feasible, alpha, lower)
        upper = torch.where(feasible, upper, alpha)
    alpha = torch.where(needs_projection, lower, torch.ones_like(lower))
    alpha = alpha.view((alpha.shape[0],) + (1,) * (mean.ndim - 1))
    return (alpha * mean, torch.exp(alpha * log_scale))

class CEMSolver:

    def __init__(self, model: Costable, batch_size: int=1, num_samples: int=300, var_scale: float=1, n_steps: int=30, topk: int=30, device: str | torch.device='cpu', seed: int=1234, callbacks: list[Callback] | None=None, bounded_actions: bool=False) -> None:
        if num_samples < 1:
            raise ValueError('num_samples must be positive')
        if not 1 <= topk <= num_samples:
            raise ValueError('topk must be in [1, num_samples]')
        if n_steps < 1:
            raise ValueError('n_steps must be positive')
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.callbacks = list(callbacks) if callbacks else []
        self.bounded_actions = bounded_actions
        self._norm_lo: torch.Tensor | None = None
        self._norm_hi: torch.Tensor | None = None
        self.record_iteration_metrics = False
        self.action_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
        self.distribution_transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True
        if not isinstance(action_space, Box):
            logging.warning(f'Action space is discrete, got {type(action_space)}. CEMSolver may not work as expected.')
        elif self.bounded_actions:
            low = np.broadcast_to(np.asarray(action_space.low, dtype=np.float64), action_space.shape)[0]
            high = np.broadcast_to(np.asarray(action_space.high, dtype=np.float64), action_space.shape)[0]
            block = getattr(config, 'action_block', 1)
            self._raw_low = torch.as_tensor(np.tile(low, block), dtype=torch.float32)
            self._raw_high = torch.as_tensor(np.tile(high, block), dtype=torch.float32)

    def set_action_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        if not self.bounded_actions:
            return
        mean = np.asarray(mean, dtype=np.float64).ravel()
        std = np.asarray(std, dtype=np.float64).ravel()
        if not hasattr(self, '_raw_low'):
            raise RuntimeError('configure() must run before set_action_normalizer()')
        n = self._raw_low.numel()
        if mean.size * (n // max(mean.size, 1)) != n:
            raise ValueError(f'normalizer stats dim {mean.size} incompatible with action dim {n}')
        reps = n // mean.size
        mean = np.tile(mean, reps)
        std = np.tile(std, reps)
        self._norm_lo = torch.as_tensor((self._raw_low.numpy() - mean) / std, dtype=torch.float32)
        self._norm_hi = torch.as_tensor((self._raw_high.numpy() - mean) / std, dtype=torch.float32)

    def _u_to_action(self, u: torch.Tensor) -> torch.Tensor:
        lo = self._norm_lo.to(device=u.device, dtype=u.dtype)
        hi = self._norm_hi.to(device=u.device, dtype=u.dtype)
        return lo + (torch.tanh(u) + 1.0) * 0.5 * (hi - lo)

    def _action_to_u(self, action: torch.Tensor) -> torch.Tensor:
        lo = self._norm_lo.to(device=action.device, dtype=action.dtype)
        hi = self._norm_hi.to(device=action.device, dtype=action.dtype)
        frac = (action.clamp(lo, hi) - lo) / (hi - lo) * 2.0 - 1.0
        return torch.atanh(frac.clamp(-0.999, 0.999))

    def _ensure_norm_bounds(self, device: torch.device, dtype: torch.dtype):
        if self._norm_lo is not None:
            self._norm_lo = self._norm_lo.to(device)
            self._norm_hi = self._norm_hi.to(device)
            return
        if not hasattr(self, '_raw_low'):
            raise RuntimeError('bounded_actions=True requires configure() with a Box space')
        logging.warning('bounded CEM without a normalizer: using raw bounds (identity normalization)')
        self._norm_lo = self._raw_low.to(device=device, dtype=dtype)
        self._norm_hi = self._raw_high.to(device=device, dtype=dtype)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_transform is None:
            return actions
        transformed = self.action_transform(actions)
        if transformed.shape != actions.shape:
            raise ValueError(f'action_transform must preserve shape, got {tuple(actions.shape)} -> {tuple(transformed.shape)}')
        return transformed

    def init_action_distrib(self, n_envs: int, actions: torch.Tensor | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        var = self.var_scale * torch.ones([n_envs, self.horizon, self.action_dim], dtype=self.dtype)
        mean = torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype) if actions is None else actions
        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([n_envs, remaining, self.action_dim], dtype=self.dtype)
            mean = torch.cat([mean, new_mean], dim=1).to(device)
        return (mean, var)

    def transform_distribution(self, mean: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.distribution_transform is None:
            return (mean, scale)
        (transformed_mean, transformed_scale) = self.distribution_transform(mean, scale)
        if transformed_mean.shape != mean.shape:
            raise ValueError('distribution transform changed mean shape')
        if transformed_scale.shape != scale.shape:
            raise ValueError('distribution transform changed scale shape')
        if torch.any(transformed_scale <= 0):
            raise ValueError('distribution transform returned nonpositive scale')
        return (transformed_mean, transformed_scale)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None=None) -> dict:
        start_time = time.time()
        outputs = {'costs': [], 'mean': [], 'var': []}
        final_elite_actions = []
        final_elite_costs = []
        total_envs = len(next(iter(info_dict.values())))
        iteration_metrics = None
        if self.record_iteration_metrics:
            iteration_metrics = {'population_mean': torch.empty(total_envs, self.n_steps), 'population_min': torch.empty(total_envs, self.n_steps), 'elite_mean': torch.empty(total_envs, self.n_steps), 'distribution_mean_kl': torch.empty(total_envs, self.n_steps)}
        init_action = prepare_init_action(self.model, info_dict, init_action, self.horizon, n_envs=total_envs, action_dim=self.action_dim)
        (mean, var) = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        if self.bounded_actions:
            self._ensure_norm_bounds(self.device, self.dtype)
            mean = self._action_to_u(mean)
        for cb in self.callbacks:
            cb.reset()
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            (batch_mean, batch_var) = self.transform_distribution(batch_mean, batch_var)
            expanded_infos = {}
            for (k, v) in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    target_dtype = self.dtype if v_batch.is_floating_point() else None
                    v_batch = v_batch.to(device=self.device, dtype=target_dtype).unsqueeze(1).expand(current_bs, self.num_samples, *v_batch.shape[1:])
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = v_batch
            final_batch_cost = None
            for cb in self.callbacks:
                cb.start_batch()
            for step in range(self.n_steps):
                candidates = torch.randn(current_bs, self.num_samples, self.horizon, self.action_dim, generator=self.torch_gen, device=self.device, dtype=self.dtype)
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean
                if self.bounded_actions:
                    evaluated_candidates = self.transform_actions(self._u_to_action(candidates))
                else:
                    evaluated_candidates = self.transform_actions(candidates)
                costs = self.model.get_cost(expanded_infos, evaluated_candidates)
                assert isinstance(costs, torch.Tensor), f'Expected cost to be a torch.Tensor, got {type(costs)}'
                assert costs.ndim == 2 and costs.shape[0] == current_bs and (costs.shape[1] == self.num_samples), f'Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}'
                (topk_vals, topk_inds) = torch.topk(costs, k=self.topk, dim=1, largest=False)
                if iteration_metrics is not None:
                    iteration_metrics['population_mean'][start_idx:end_idx, step] = costs.mean(dim=1).detach().cpu()
                    iteration_metrics['population_min'][start_idx:end_idx, step] = topk_vals[:, 0].detach().cpu()
                    iteration_metrics['elite_mean'][start_idx:end_idx, step] = topk_vals.mean(dim=1).detach().cpu()
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                evaluated_topk_candidates = evaluated_candidates[batch_indices, topk_inds]
                prev_mean = batch_mean
                prev_var = batch_var
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)
                (batch_mean, batch_var) = self.transform_distribution(batch_mean, batch_var)
                if iteration_metrics is not None:
                    iteration_metrics['distribution_mean_kl'][start_idx:end_idx, step] = standard_normal_kl(batch_mean, batch_var).detach().cpu()
                for cb in self.callbacks:
                    cb(step=step, candidates=evaluated_candidates, costs=costs, topk_vals=topk_vals, topk_inds=topk_inds, topk_candidates=evaluated_topk_candidates, mean=self.transform_actions(self._u_to_action(batch_mean) if self.bounded_actions else batch_mean), var=batch_var, prev_mean=self.transform_actions(self._u_to_action(prev_mean) if self.bounded_actions else prev_mean), prev_var=prev_var)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs['costs'].extend(final_batch_cost)
            final_elite_actions.append(evaluated_topk_candidates.detach().cpu())
            final_elite_costs.append(topk_vals.detach().cpu())
        if self.bounded_actions:
            mean = self._u_to_action(mean)
        transformed_mean = self.transform_actions(mean)
        outputs['actions'] = transformed_mean.detach().cpu()
        outputs['mean'] = [transformed_mean.detach().cpu()]
        outputs['var'] = [var.detach().cpu()]
        outputs['elite_actions'] = torch.cat(final_elite_actions, dim=0)
        outputs['elite_costs'] = torch.cat(final_elite_costs, dim=0)
        if iteration_metrics is not None:
            outputs['iteration_metrics'] = {key: value.tolist() for (key, value) in iteration_metrics.items()}
        if self.callbacks:
            outputs['callbacks'] = {}
            for cb in self.callbacks:
                cb.end_solve()
                outputs['callbacks'][cb.output_key] = cb.history
        outputs['solve_time'] = time.time() - start_time
        return outputs
