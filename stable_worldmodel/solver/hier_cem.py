import time
from collections import deque
from functools import partial
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging
from stable_worldmodel.solver.cem import CEMSolver, project_standard_normal_kl
from stable_worldmodel.solver.categorical_cem import CategoricalCEMSolver
from stable_worldmodel.solver.spaces import is_discrete_space
from stable_worldmodel.solver.utils import prepare_init_action

class HierarchicalCEMSolver:

    def __init__(self, model, embed_dim: int, u_dim: int=16, u_num_samples: int=100, u_n_steps: int=20, u_topk: int=10, u_horizon: int=6, u_prior_constraint: str='none', u_prior_kl_limit: float=0.1, subgoal_rerank_topk: int=1, subgoal_rerank_num_samples: int=64, subgoal_rerank_n_steps: int=5, subgoal_rerank_elite: int=8, subgoal_rerank_weight: float=0.5, subgoal_rerank_normalization: str='minmax', subgoal_rerank_refine_steps: int=5, subgoal_rerank_reuse_actions: bool=True, a_num_samples: int=300, a_n_steps: int=30, a_topk: int=30, max_high_level_replans: int=1, high_level_guidance_steps: int | None=None, high_level_guidance_macro_steps: int | None=None, high_level_replan_interval: int=1, high_level_start_context: str='repeat_current', allow_padded_high_level_start: bool=True, final_after_macro_complete: bool=False, reset_low_level_warm_start_on_objective_change: bool=True, a_smoothing: float=0.1, a_alpha: float=0.5, a_warm_start_escape: float | None=None, low_level_discrete: bool | None=None, final_receding_horizon: int=1, batch_size: int=1, var_scale: float=1.0, device: str | torch.device='cuda', seed: int=1234, a_bounded_actions: bool=False, a_var_scale: float | None=None) -> None:
        self.model = model
        self.embed_dim = embed_dim
        self.u_dim = u_dim
        self.u_horizon = u_horizon
        if u_prior_constraint not in ('none', 'kl_trust_region'):
            raise ValueError(f'u_prior_constraint must be none|kl_trust_region, got {u_prior_constraint!r}')
        if u_prior_kl_limit < 0:
            raise ValueError('u_prior_kl_limit must be non-negative')
        self.u_prior_constraint = u_prior_constraint
        self.u_prior_kl_limit = float(u_prior_kl_limit)
        if subgoal_rerank_topk < 1:
            raise ValueError('subgoal_rerank_topk must be at least 1')
        if subgoal_rerank_topk > u_topk + 1:
            raise ValueError('subgoal_rerank_topk cannot exceed u_topk + 1 because the candidate set contains the CEM mean plus final elites')
        if subgoal_rerank_num_samples < 2:
            raise ValueError('subgoal_rerank_num_samples must be at least 2')
        if not 1 <= subgoal_rerank_elite <= subgoal_rerank_num_samples:
            raise ValueError('subgoal_rerank_elite must be in [1, subgoal_rerank_num_samples]')
        if subgoal_rerank_n_steps < 1:
            raise ValueError('subgoal_rerank_n_steps must be at least 1')
        if subgoal_rerank_topk > 1 and (not 0 <= subgoal_rerank_refine_steps <= a_n_steps):
            raise ValueError('subgoal_rerank_refine_steps must be in [0, a_n_steps]')
        if subgoal_rerank_weight < 0:
            raise ValueError('subgoal_rerank_weight must be non-negative')
        if subgoal_rerank_normalization not in ('minmax', 'raw'):
            raise ValueError('subgoal_rerank_normalization must be minmax|raw')
        self.subgoal_rerank_topk = int(subgoal_rerank_topk)
        self.subgoal_rerank_weight = float(subgoal_rerank_weight)
        self.subgoal_rerank_normalization = subgoal_rerank_normalization
        self.subgoal_rerank_refine_steps = int(subgoal_rerank_refine_steps)
        self.subgoal_rerank_num_samples = int(subgoal_rerank_num_samples)
        self.subgoal_rerank_n_steps = int(subgoal_rerank_n_steps)
        self.subgoal_rerank_elite = int(subgoal_rerank_elite)
        self.subgoal_rerank_reuse_actions = bool(subgoal_rerank_reuse_actions)
        self.max_high_level_replans = max_high_level_replans
        if not 1 <= high_level_replan_interval <= u_horizon:
            raise ValueError(f'high_level_replan_interval must be in [1, u_horizon], got {high_level_replan_interval} for u_horizon={u_horizon}')
        self.high_level_replan_interval = int(high_level_replan_interval)
        if high_level_guidance_steps is not None and high_level_guidance_macro_steps is not None:
            raise ValueError('Specify only one of high_level_guidance_steps (legacy low-level decisions) and high_level_guidance_macro_steps (complete high-level transitions)')
        if high_level_guidance_steps is not None:
            if high_level_guidance_steps < 0:
                raise ValueError('high_level_guidance_steps must be non-negative')
            if max_high_level_replans > 0 and high_level_guidance_steps > max_high_level_replans * model.window_size:
                raise ValueError('high_level_guidance_steps cannot exceed max_high_level_replans * window_size')
        self.high_level_guidance_steps = high_level_guidance_steps
        if high_level_guidance_macro_steps is not None:
            if high_level_guidance_macro_steps < 0:
                raise ValueError('high_level_guidance_macro_steps must be non-negative')
            if max_high_level_replans > 0 and high_level_guidance_macro_steps > max_high_level_replans * self.high_level_replan_interval:
                raise ValueError('high_level_guidance_macro_steps cannot exceed max_high_level_replans * high_level_replan_interval')
        self.high_level_guidance_macro_steps = high_level_guidance_macro_steps
        if high_level_start_context not in ('zero', 'repeat_current'):
            raise ValueError(f'high_level_start_context must be zero|repeat_current, got {high_level_start_context!r}')
        self.high_level_start_context = high_level_start_context
        self.allow_padded_high_level_start = allow_padded_high_level_start
        self.final_after_macro_complete = final_after_macro_complete
        self.reset_low_level_warm_start_on_objective_change = bool(reset_low_level_warm_start_on_objective_change)
        self.final_receding_horizon = int(final_receding_horizon)
        if self.final_receding_horizon < 1:
            raise ValueError('final_receding_horizon must be >= 1')
        self.a_num_samples = int(a_num_samples)
        self.a_n_steps = int(a_n_steps)
        self.a_topk = int(a_topk)
        self.a_smoothing = float(a_smoothing)
        self.a_alpha = float(a_alpha)
        self.a_warm_start_escape = a_warm_start_escape
        self.batch_size = int(batch_size)
        self.var_scale = float(var_scale)
        self.a_bounded_actions = bool(a_bounded_actions)
        self.a_var_scale = None if a_var_scale is None else float(a_var_scale)
        if self.a_var_scale is not None and (not np.isfinite(self.a_var_scale) or self.a_var_scale <= 0):
            raise ValueError('a_var_scale must be finite and positive')
        self.seed = int(seed)
        self.low_level_discrete = low_level_discrete
        self._discrete_low_level = False
        self._low_action_dim: int | None = None
        self.k = model.window_size
        self.HS_H = model.high_level_history_size
        self.device = torch.device(device)
        self.solver_u = CEMSolver(model=model, num_samples=u_num_samples, n_steps=u_n_steps, topk=u_topk, batch_size=batch_size, var_scale=var_scale, device=device, seed=seed)
        if u_prior_constraint == 'kl_trust_region':
            self.solver_u.distribution_transform = partial(project_standard_normal_kl, max_mean_kl=self.u_prior_kl_limit)
        self.solver_a = CEMSolver(model=model, num_samples=a_num_samples, n_steps=a_n_steps, topk=a_topk, batch_size=batch_size, var_scale=self.var_scale if self.a_var_scale is None else self.a_var_scale, bounded_actions=self.a_bounded_actions, device=device, seed=seed)
        self.solver_a_feasibility = CEMSolver(model=model, num_samples=subgoal_rerank_num_samples, n_steps=subgoal_rerank_n_steps, topk=subgoal_rerank_elite, batch_size=max(1, batch_size * subgoal_rerank_topk), var_scale=self.var_scale if self.a_var_scale is None else self.a_var_scale, bounded_actions=self.a_bounded_actions, device=device, seed=seed + 1)
        self._z_l_history: list[deque] | None = None
        self._z_h_history: list[deque] | None = None
        self._current_z_h_subgoal: list[torch.Tensor | None] | None = None
        self._pending_z_h_subgoals: list[deque] | None = None
        self._macro_progress: list[int] | None = None
        self._high_level_guidance_progress: list[int] | None = None
        self._high_level_guidance_macro_progress: list[int] | None = None
        self._high_level_replan_count: list[int] | None = None
        self._high_level_subgoal_assignment_count: list[int] | None = None
        self._goal_emb: list[torch.Tensor | None] | None = None
        self._current_z_l_subgoal: list[torch.Tensor | None] | None = None
        self._last_low_level_objective: list[tuple[str, int] | None] | None = None
        self._subgoal_completion_recorded: list[bool] | None = None
        self._current_subgoal_rerank_meta: list[dict | None] | None = None
        self._final_plan_cache: list[torch.Tensor | None] | None = None
        self._final_plan_consumed: list[int] | None = None

    @property
    def action_dim(self) -> int:
        if self._low_action_dim is not None:
            return self._low_action_dim
        return self.solver_a.action_dim

    @property
    def n_envs(self) -> int:
        return self.solver_a.n_envs

    @property
    def horizon(self) -> int:
        return self.solver_a.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def _make_low_level_solver(self, *, discrete: bool, num_samples: int, n_steps: int, topk: int, batch_size: int, seed: int):
        common = dict(model=self.model, num_samples=num_samples, n_steps=n_steps, topk=topk, batch_size=batch_size, device=self.device, seed=seed)
        if discrete:
            return CategoricalCEMSolver(**common, smoothing=self.a_smoothing, alpha=self.a_alpha, warm_start_escape=self.a_warm_start_escape)
        return CEMSolver(**common, var_scale=self.var_scale if self.a_var_scale is None else self.a_var_scale, bounded_actions=self.a_bounded_actions)

    def set_action_normalizer(self, mean, std):
        for solver in [self.solver_a, self.solver_a_feasibility]:
            if isinstance(solver, CEMSolver):
                solver.set_action_normalizer(mean, std)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        from stable_worldmodel.policy import PlanConfig
        if self.max_high_level_replans > 0:
            assert config.receding_horizon == 1, 'Dual-WM first version requires receding_horizon=1 so that z_l_history is updated at every low-level model step.'
        macro_space = Box(low=-5.0, high=5.0, shape=(n_envs, self.u_dim), dtype=np.float32)
        u_config = PlanConfig(horizon=self.u_horizon, receding_horizon=1, action_block=1)
        self.solver_u.configure(action_space=macro_space, n_envs=n_envs, config=u_config)
        a_config = PlanConfig(horizon=config.horizon, receding_horizon=config.receding_horizon, action_block=config.action_block, warm_start=config.warm_start)
        is_discrete = is_discrete_space(action_space)
        use_categorical = is_discrete if self.low_level_discrete is None else bool(self.low_level_discrete)
        if use_categorical and (not is_discrete):
            raise ValueError(f'low_level_discrete=True requires a Discrete action space, got {type(action_space).__name__}')
        if is_discrete and (not use_categorical):
            raise ValueError('low_level_discrete=False cannot be used with a discrete environment: Gaussian CEM would optimize invalid continuous action values. Use null/true, or a continuous environment.')
        if use_categorical and (not self._discrete_low_level):
            self.solver_a = self._make_low_level_solver(discrete=True, num_samples=self.a_num_samples, n_steps=self.a_n_steps, topk=self.a_topk, batch_size=self.batch_size, seed=self.seed)
            self.solver_a_feasibility = self._make_low_level_solver(discrete=True, num_samples=self.subgoal_rerank_num_samples, n_steps=self.subgoal_rerank_n_steps, topk=self.subgoal_rerank_elite, batch_size=max(1, self.batch_size * self.subgoal_rerank_topk), seed=self.seed + 1)
            self._discrete_low_level = True
            logging.info('[hier_cem] Discrete action space -> low-level planning uses CategoricalCEMSolver (one-hot candidates).')
        elif self._discrete_low_level and (not use_categorical):
            self.solver_a = self._make_low_level_solver(discrete=False, num_samples=self.a_num_samples, n_steps=self.a_n_steps, topk=self.a_topk, batch_size=self.batch_size, seed=self.seed)
            self.solver_a_feasibility = self._make_low_level_solver(discrete=False, num_samples=self.subgoal_rerank_num_samples, n_steps=self.subgoal_rerank_n_steps, topk=self.subgoal_rerank_elite, batch_size=max(1, self.batch_size * self.subgoal_rerank_topk), seed=self.seed + 1)
            self._discrete_low_level = False
        self.solver_a.configure(action_space=action_space, n_envs=n_envs, config=a_config)
        feasibility_action_space = action_space
        if self._discrete_low_level and self.subgoal_rerank_topk > 1:
            feasibility_action_space = gym.spaces.Discrete(self.solver_a.base_simplex_dim)
        self.solver_a_feasibility.configure(action_space=feasibility_action_space, n_envs=n_envs * self.subgoal_rerank_topk, config=a_config)
        self._low_action_dim = int(a_config.action_block) if self._discrete_low_level else int(self.solver_a.action_dim)
        if self._low_action_dim < 1:
            raise ValueError(f'Invalid low-level action width {self._low_action_dim}')
        self._z_l_history = [deque(maxlen=self.k) for _ in range(n_envs)]
        self._z_h_history = [deque(maxlen=max(0, self.HS_H - 1)) for _ in range(n_envs)]
        self._current_z_h_subgoal = [None] * n_envs
        self._pending_z_h_subgoals = [deque() for _ in range(n_envs)]
        self._macro_progress = [0] * n_envs
        self._high_level_guidance_progress = [0] * n_envs
        self._high_level_guidance_macro_progress = [0] * n_envs
        self._high_level_replan_count = [0] * n_envs
        self._high_level_subgoal_assignment_count = [0] * n_envs
        self._goal_emb = [None] * n_envs
        self._current_z_l_subgoal = [None] * n_envs
        self._last_low_level_objective = [None] * n_envs
        self._subgoal_completion_recorded = [False] * n_envs
        self._current_subgoal_rerank_meta = [None] * n_envs
        self._final_plan_cache = [None] * n_envs
        self._final_plan_consumed = [0] * n_envs

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        p = next(self.model.parameters())
        return (p.device, p.dtype)

    def _to_model_tensor(self, x: Any) -> Any:
        (device, dtype) = self._model_device_dtype()
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype if x.is_floating_point() else None)
        return x

    def _select_model_inputs(self, info_dict: dict) -> dict:
        allowed = {'pixels'}
        out = {}
        for (k, v) in info_dict.items():
            if k in allowed or k.startswith('goal_'):
                out[k] = self._to_model_tensor(v)
        return out

    def _encode_current(self, info_dict: dict) -> torch.Tensor:
        enc_info = self._select_model_inputs(info_dict)
        for (k, v) in list(enc_info.items()):
            if k == 'pixels' and torch.is_tensor(v):
                if v.ndim == 5:
                    enc_info[k] = v[:, :1]
                elif v.ndim == 4:
                    enc_info[k] = v.unsqueeze(1)
        output = self.model.encode(enc_info)
        return output['emb'][:, -1]

    def _get_goal_emb(self, info_dict: dict, env_indices: torch.Tensor) -> torch.Tensor:
        (device, dtype) = self._model_device_dtype()
        goal_embs = []
        for (row, env_i) in enumerate(env_indices.tolist()):
            if self._goal_emb[env_i] is None:
                goal_pixels = info_dict['goal'][row:row + 1]
                goal_pixels = goal_pixels.to(device=device, dtype=dtype)
                if goal_pixels.ndim == 4:
                    goal_pixels = goal_pixels.unsqueeze(1)
                self._goal_emb[env_i] = self.model.encode({'pixels': goal_pixels})['emb'][:, -1]
            goal_embs.append(self._goal_emb[env_i].squeeze(0))
        return torch.stack(goal_embs, dim=0)

    def _build_high_level_state(self, cur_z_l: torch.Tensor, env_indices: torch.Tensor) -> torch.Tensor:
        z_l_windows = []
        for (row, env_i) in enumerate(env_indices.tolist()):
            hist = list(self._z_l_history[env_i])
            hist.append(cur_z_l[row])
            if len(hist) < self.k:
                if self.high_level_start_context == 'repeat_current':
                    pad_value = hist[0]
                else:
                    pad_value = hist[0].new_zeros(hist[0].shape)
                pad = [pad_value] * (self.k - len(hist))
                hist = pad + hist
            else:
                hist = hist[-self.k:]
            z_l_windows.append(torch.stack(hist, dim=0))
        z_l_window = torch.stack(z_l_windows, dim=0)
        return self.model.temporal_encoder(z_l_window)

    def _build_high_level_predictor_context(self, z_h_current: torch.Tensor, env_indices: torch.Tensor) -> torch.Tensor:
        ctx_list = []
        for (row, env_i) in enumerate(env_indices.tolist()):
            hist = list(self._z_h_history[env_i])
            hist.append(z_h_current[row])
            if len(hist) < self.HS_H:
                if self.high_level_start_context == 'repeat_current' and len(hist) == 1:
                    pad_value = z_h_current[row]
                else:
                    pad_value = z_h_current[row].new_zeros(z_h_current[row].shape)
                pad = [pad_value] * (self.HS_H - len(hist))
                hist = pad + hist
            else:
                hist = hist[-self.HS_H:]
            ctx_list.append(torch.stack(hist, dim=0))
        return torch.stack(ctx_list, dim=0)

    def _flush_if_needed(self, env_indices: torch.Tensor, needs_flush: torch.Tensor | None) -> None:
        if needs_flush is None:
            return
        for (row, env_i) in enumerate(env_indices.tolist()):
            if bool(needs_flush[row]):
                self._z_l_history[env_i].clear()
                self._z_h_history[env_i].clear()
                self._current_z_h_subgoal[env_i] = None
                self._pending_z_h_subgoals[env_i].clear()
                self._current_z_l_subgoal[env_i] = None
                self._macro_progress[env_i] = 0
                self._high_level_guidance_progress[env_i] = 0
                self._high_level_guidance_macro_progress[env_i] = 0
                self._high_level_replan_count[env_i] = 0
                self._high_level_subgoal_assignment_count[env_i] = 0
                self._goal_emb[env_i] = None
                self._last_low_level_objective[env_i] = None
                self._subgoal_completion_recorded[env_i] = False
                self._current_subgoal_rerank_meta[env_i] = None
                if self._final_plan_cache is not None:
                    self._final_plan_cache[env_i] = None
                    self._final_plan_consumed[env_i] = 0

    def _prepare_low_level_warm_start(self, clean_info: dict, init_action: torch.Tensor | None, env_indices: torch.Tensor, objective_keys: list[tuple[str, int]]) -> tuple[torch.Tensor | None, list[bool]]:
        if len(objective_keys) != len(env_indices):
            raise ValueError('objective_keys and env_indices must have the same length')
        reset_mask = []
        for (env_i, objective_key) in zip(env_indices.detach().cpu().tolist(), objective_keys):
            previous = self._last_low_level_objective[env_i]
            objective_changed = previous is not None and previous != objective_key
            final_plan_cache = getattr(self, '_final_plan_cache', None)
            if objective_changed and final_plan_cache is not None:
                final_plan_cache[env_i] = None
                self._final_plan_consumed[env_i] = 0
            reset_mask.append(bool(self.reset_low_level_warm_start_on_objective_change and init_action is not None and objective_changed))
            self._last_low_level_objective[env_i] = objective_key
        if init_action is None or not any(reset_mask):
            return (init_action, reset_mask)
        if all(reset_mask):
            return (None, reset_mask)
        configured_dim = getattr(self, '_low_action_dim', None)
        action_dim = configured_dim if configured_dim is not None else self.solver_a.action_dim
        reset_idx = torch.as_tensor([i for (i, reset) in enumerate(reset_mask) if reset], device=env_indices.device, dtype=torch.long)
        keep_idx = torch.as_tensor([i for (i, reset) in enumerate(reset_mask) if not reset], device=env_indices.device, dtype=torch.long)
        if getattr(self, '_discrete_low_level', False):
            prepared = torch.full((len(env_indices), self.solver_a.horizon, action_dim), -1, device=init_action.device, dtype=init_action.dtype)
            steps = min(init_action.shape[1], self.solver_a.horizon)
            prepared[keep_idx.to(prepared.device), :steps] = init_action[keep_idx.to(init_action.device), :steps].to(prepared.device)
            return (prepared, reset_mask)
        keep_info = self._slice_info(clean_info, keep_idx)
        reset_info = self._slice_info(clean_info, reset_idx)
        prepared_keep = prepare_init_action(self.model, keep_info, init_action[keep_idx], self.solver_a.horizon, n_envs=len(keep_idx), action_dim=action_dim)
        prepared_reset = prepare_init_action(self.model, reset_info, None, self.solver_a.horizon, n_envs=len(reset_idx), action_dim=action_dim)
        prepared = torch.empty(len(env_indices), self.solver_a.horizon, action_dim, device=prepared_keep.device, dtype=prepared_keep.dtype)
        prepared[keep_idx.to(prepared.device)] = prepared_keep
        prepared[reset_idx.to(prepared.device)] = prepared_reset.to(device=prepared.device, dtype=prepared.dtype)
        return (prepared, reset_mask)

    def _update_histories(self, env_indices: torch.Tensor, cur_z_l: torch.Tensor, z_h_current: torch.Tensor) -> None:
        for (row, env_i) in enumerate(env_indices.tolist()):
            high_level_ready = len(self._z_l_history[env_i]) + 1 >= self.k
            self._z_l_history[env_i].append(cur_z_l[row].detach())
            if high_level_ready:
                self._z_h_history[env_i].append(z_h_current[row].detach())

    def _high_level_ready_mask(self, env_indices: torch.Tensor) -> torch.Tensor:
        return torch.tensor([len(self._z_l_history[env_i]) + 1 >= self.k for env_i in env_indices.tolist()], device=env_indices.device, dtype=torch.bool)

    def _should_enter_final(self, env_i: int) -> bool:
        if self.max_high_level_replans <= 0:
            return True
        guidance_macro_steps = getattr(self, 'high_level_guidance_macro_steps', None)
        if guidance_macro_steps is not None:
            progress = self._high_level_guidance_macro_progress[env_i]
            return progress >= guidance_macro_steps
        guidance_steps = getattr(self, 'high_level_guidance_steps', None)
        if guidance_steps is not None:
            progress = self._high_level_guidance_progress[env_i]
            return progress >= guidance_steps
        if self._high_level_replan_count[env_i] < self.max_high_level_replans:
            return False
        if not self.final_after_macro_complete:
            return True
        if self._current_z_h_subgoal[env_i] is None and self._current_z_l_subgoal[env_i] is None:
            return True
        return self._macro_progress[env_i] >= self.k

    def _advance_subgoal_progress(self, env_indices: torch.Tensor) -> None:
        for env_i in env_indices.tolist():
            previous = self._macro_progress[env_i]
            self._macro_progress[env_i] += 1
            self._high_level_guidance_progress[env_i] += 1
            if previous < self.k and self._macro_progress[env_i] >= self.k:
                self._high_level_guidance_macro_progress[env_i] += 1

    def _remaining_high_level_commitment(self, env_i: int) -> int:
        commitment = self.high_level_replan_interval
        guidance_macro_steps = self.high_level_guidance_macro_steps
        if guidance_macro_steps is None:
            return commitment
        remaining = guidance_macro_steps - self._high_level_guidance_macro_progress[env_i]
        return max(0, min(commitment, remaining))

    def _assign_planned_high_level_subgoals(self, env_indices: torch.Tensor, need_high_replan: list[bool], z_h_plan: list[torch.Tensor], u_outputs: dict) -> None:
        if len(z_h_plan) < self.high_level_replan_interval:
            raise ValueError('High-level rollout returned fewer states than high_level_replan_interval')
        rerank = u_outputs.get('rerank')
        for (row, env_i) in enumerate(env_indices.tolist()):
            if not need_high_replan[row]:
                continue
            commitment = self._remaining_high_level_commitment(env_i)
            if commitment < 1:
                raise RuntimeError('Attempted a high-level replan after its guidance budget was exhausted')
            self._current_z_h_subgoal[env_i] = z_h_plan[0][row].detach()
            self._current_z_l_subgoal[env_i] = None
            pending = self._pending_z_h_subgoals[env_i]
            pending.clear()
            for macro_idx in range(1, commitment):
                pending.append(z_h_plan[macro_idx][row].detach())
            self._macro_progress[env_i] = 0
            self._high_level_replan_count[env_i] += 1
            self._high_level_subgoal_assignment_count[env_i] += 1
            self._subgoal_completion_recorded[env_i] = False
            if rerank is None:
                self._current_subgoal_rerank_meta[env_i] = None
            else:
                selected = int(rerank['selected_candidate'][row])
                self._current_subgoal_rerank_meta[env_i] = {'selected_candidate': selected, 'predicted_feasibility_cost': float(rerank['candidate_feasibility_cost'][row, selected])}

    def _activate_cached_or_request_replan(self, env_indices: torch.Tensor) -> list[bool]:
        need_high_replan = []
        for env_i in env_indices.tolist():
            current = self._current_z_h_subgoal[env_i]
            if current is None:
                need_high_replan.append(True)
                continue
            if self._macro_progress[env_i] < self.k:
                need_high_replan.append(False)
                continue
            pending = self._pending_z_h_subgoals[env_i]
            if current is not None and pending:
                self._current_z_h_subgoal[env_i] = pending.popleft()
                self._current_z_l_subgoal[env_i] = None
                self._macro_progress[env_i] = 0
                self._high_level_subgoal_assignment_count[env_i] += 1
                self._subgoal_completion_recorded[env_i] = False
                self._current_subgoal_rerank_meta[env_i] = None
                need_high_replan.append(False)
            else:
                need_high_replan.append(True)
        return need_high_replan

    def _build_subgoal_prefix(self, env_indices: torch.Tensor, cur_z_l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prefixes = []
        lengths = []
        for (row, env_i) in enumerate(env_indices.tolist()):
            progress = int(self._macro_progress[env_i])
            if not 0 <= progress < self.k:
                raise ValueError(f'macro progress must be in [0,{self.k - 1}] while tracking a subgoal, got {progress}')
            available = list(self._z_l_history[env_i]) + [cur_z_l[row]]
            observed = available[-progress:] if progress else []
            if len(observed) != progress:
                raise ValueError(f'Need {progress} observed prefix states, got {len(observed)}')
            padding = [torch.zeros_like(cur_z_l[row])] * (self.k - 1 - progress)
            prefixes.append(torch.stack(observed + padding, dim=0))
            lengths.append(progress)
        return (torch.stack(prefixes, dim=0), torch.as_tensor(lengths, device=cur_z_l.device, dtype=torch.long))

    def _build_final_mask(self, env_indices: torch.Tensor) -> torch.Tensor:
        return torch.tensor([self._should_enter_final(i.item()) for i in env_indices], device=env_indices.device, dtype=torch.bool)

    @staticmethod
    def _normalize_rerank_cost(cost: torch.Tensor) -> torch.Tensor:
        minimum = cost.min(dim=1, keepdim=True).values
        span = cost.max(dim=1, keepdim=True).values - minimum
        eps = torch.finfo(cost.dtype).eps
        return torch.where(span > eps, (cost - minimum) / span.clamp_min(eps), torch.zeros_like(cost))

    @staticmethod
    def _expand_candidate_axis(info_dict: dict, count: int) -> dict:
        expanded = {}
        for (key, value) in info_dict.items():
            if torch.is_tensor(value):
                expanded[key] = value.unsqueeze(1).expand(value.shape[0], count, *value.shape[1:])
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value[:, None], count, axis=1)
            elif isinstance(value, list):
                expanded[key] = [[item] * count for item in value]
            else:
                expanded[key] = value
        return expanded

    @staticmethod
    def _repeat_candidate_rows(info_dict: dict, count: int) -> dict:
        repeated = {}
        for (key, value) in info_dict.items():
            if torch.is_tensor(value):
                repeated[key] = value.repeat_interleave(count, dim=0)
            elif isinstance(value, np.ndarray):
                repeated[key] = np.repeat(value, count, axis=0)
            elif isinstance(value, list):
                repeated[key] = [item for item in value for _ in range(count)]
            else:
                repeated[key] = value
        return repeated

    def _rerank_high_level_candidates(self, clean_info: dict, z_h_context: torch.Tensor, u_outputs: dict) -> tuple[list[torch.Tensor], dict, torch.Tensor]:
        count = self.subgoal_rerank_topk
        if count <= 1:
            raise RuntimeError('reranking requires at least two candidates')
        (device, dtype) = self._model_device_dtype()
        mean_action = u_outputs['actions'].to(device=device, dtype=dtype)
        elite_action = u_outputs['elite_actions'][:, :count - 1].to(device=device, dtype=dtype)
        u_candidates = torch.cat([mean_action.unsqueeze(1), elite_action], dim=1)
        batch_size = u_candidates.shape[0]
        high_info = self._expand_candidate_axis(clean_info, count)
        self.model._cost_mode = 'high_level'
        try:
            high_cost = self.model.get_cost(high_info, u_candidates)
        finally:
            self.model._cost_mode = 'default'
        z_h_candidates = self.model.rollout_high_level_from_context(z_h_context, u_candidates)
        first_subgoal = z_h_candidates[0]
        low_info = self._repeat_candidate_rows(clean_info, count)
        low_info['z_h_subgoal'] = first_subgoal.reshape(batch_size * count, self.embed_dim)
        low_info['z_l_subgoal_prefix'] = first_subgoal.new_zeros(batch_size * count, max(0, self.k - 1), self.embed_dim)
        low_info['z_l_subgoal_prefix_len'] = torch.zeros(batch_size * count, device=first_subgoal.device, dtype=torch.long)
        self.model._cost_mode = 'low_level_subgoal'
        try:
            feasibility_outputs = self.solver_a_feasibility(low_info)
        finally:
            self.model._cost_mode = 'default'
        feasibility_cost = torch.as_tensor(feasibility_outputs['costs'], device=device, dtype=dtype).reshape(batch_size, count)
        if self.subgoal_rerank_normalization == 'minmax':
            high_score = self._normalize_rerank_cost(high_cost)
            feasibility_score = self._normalize_rerank_cost(feasibility_cost)
        else:
            high_score = high_cost
            feasibility_score = feasibility_cost
        combined_score = high_score + self.subgoal_rerank_weight * feasibility_score
        selected = combined_score.argmin(dim=1)
        batch_index = torch.arange(batch_size, device=device)
        selected_plan = [state[batch_index, selected] for state in z_h_candidates]
        selected_u = u_candidates[batch_index, selected]
        feasibility_actions = feasibility_outputs['actions'].to(device=device, dtype=torch.long if self._discrete_low_level else dtype).reshape(batch_size, count, self.solver_a_feasibility.horizon, self._low_action_dim)
        selected_low_action = feasibility_actions[batch_index, selected]
        selected_outputs = dict(u_outputs)
        selected_outputs['actions'] = selected_u.detach().cpu()
        selected_outputs['costs'] = high_cost[batch_index, selected].detach().cpu().tolist()
        selected_outputs['rerank'] = {'candidate_high_cost': high_cost.detach().cpu(), 'candidate_feasibility_cost': feasibility_cost.detach().cpu(), 'candidate_combined_score': combined_score.detach().cpu(), 'selected_candidate': selected.detach().cpu(), 'feasibility_solve_time': float(feasibility_outputs['solve_time'])}
        return (selected_plan, selected_outputs, selected_low_action)

    def _plan_high_level(self, clean_info: dict, z_h_context: torch.Tensor) -> tuple[list[torch.Tensor], dict, torch.Tensor | None]:
        self.model._cost_mode = 'high_level'
        try:
            u_outputs = self.solver_u(clean_info)
        finally:
            self.model._cost_mode = 'default'
        if self.subgoal_rerank_topk > 1:
            return self._rerank_high_level_candidates(clean_info, z_h_context, u_outputs)
        u_best = u_outputs['actions'].to(device=z_h_context.device, dtype=z_h_context.dtype)
        return (self.model.rollout_high_level_from_context(z_h_context, u_best), u_outputs, None)

    def _merge_rerank_warm_start(self, init_action: torch.Tensor | None, selected_action: torch.Tensor | None, need_high_replan: list[bool], clean_info: dict) -> torch.Tensor | None:
        if selected_action is None or not self.subgoal_rerank_reuse_actions:
            return init_action
        row_mask = torch.as_tensor(need_high_replan, device=selected_action.device, dtype=torch.bool)
        if bool(row_mask.all()):
            return selected_action
        if init_action is None:
            if getattr(self, '_discrete_low_level', False):
                init_action = torch.full_like(selected_action, -1)
            else:
                init_action = prepare_init_action(self.model, clean_info, None, self.solver_a.horizon, n_envs=len(need_high_replan), action_dim=self._low_action_dim)
        merged = init_action.to(device=selected_action.device, dtype=selected_action.dtype).clone()
        merged[row_mask] = selected_action[row_mask]
        return merged

    def _solve_low_level(self, clean_info: dict, init_action: torch.Tensor | None, use_short_refinement: bool) -> dict:
        if not use_short_refinement:
            return self.solver_a(clean_info, init_action=init_action)
        original_steps = self.solver_a.n_steps
        self.solver_a.n_steps = self.subgoal_rerank_refine_steps
        try:
            if self.solver_a.n_steps == 0:
                return {'actions': init_action.detach().cpu(), 'costs': [float('nan')] * len(init_action), 'solve_time': 0.0}
            return self.solver_a(clean_info, init_action=init_action)
        finally:
            self.solver_a.n_steps = original_steps

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None=None) -> dict:
        start_time = time.time()
        env_indices = info_dict['_env_indices']
        needs_flush = info_dict.get('_needs_flush')
        self._flush_if_needed(env_indices, needs_flush)
        cur_z_l = self._encode_current(info_dict)
        z_h_current = self._build_high_level_state(cur_z_l, env_indices)
        goal_emb = self._get_goal_emb(info_dict, env_indices)
        info_dict['goal_emb'] = goal_emb
        final_mask = self._build_final_mask(env_indices).to(goal_emb.device)
        high_level_ready = self._high_level_ready_mask(env_indices).to(goal_emb.device)
        if self.allow_padded_high_level_start:
            high_level_ready = torch.ones_like(high_level_ready)
        final_mask = final_mask | ~high_level_ready
        if final_mask.any() and (not final_mask.all()):
            return self._solve_split_by_final_mask(info_dict, init_action, final_mask, env_indices, cur_z_l, z_h_current, start_time)
        if final_mask.all():
            self.model._cost_mode = 'final_goal'
            clean_info = self._strip_metadata(info_dict)
            (init_action, warm_start_reset) = self._prepare_low_level_warm_start(clean_info, init_action, env_indices, [('final_goal', 0)] * len(env_indices))
            outputs = self._final_goal_solve_with_cache(clean_info, init_action, env_indices)
            self.model._cost_mode = 'default'
            self._update_histories(env_indices, cur_z_l, z_h_current)
            outputs['solve_time'] = time.time() - start_time
            return outputs
        need_high_replan = self._activate_cached_or_request_replan(env_indices)
        rerank_init_action = None
        if any(need_high_replan):
            z_h_ctx = self._build_high_level_predictor_context(z_h_current, env_indices)
            info_dict['z_h_context'] = z_h_ctx
            clean_info = self._strip_metadata(info_dict)
            (z_h_plan, u_outputs, rerank_init_action) = self._plan_high_level(clean_info, z_h_ctx)
            self._assign_planned_high_level_subgoals(env_indices, need_high_replan, z_h_plan, u_outputs)
        z_h_subgoal = []
        for env_i in env_indices.tolist():
            z_h_subgoal.append(self._current_z_h_subgoal[env_i])
        info_dict['z_h_subgoal'] = torch.stack(z_h_subgoal, dim=0)
        (prefix, prefix_len) = self._build_subgoal_prefix(env_indices, cur_z_l)
        info_dict['z_l_subgoal_prefix'] = prefix
        info_dict['z_l_subgoal_prefix_len'] = prefix_len
        self.model._cost_mode = 'low_level_subgoal'
        clean_info = self._strip_metadata(info_dict)
        objective_type = 'high_subgoal'
        objective_keys = [(objective_type, self._high_level_subgoal_assignment_count[env_i]) for env_i in env_indices.detach().cpu().tolist()]
        (init_action, warm_start_reset) = self._prepare_low_level_warm_start(clean_info, init_action, env_indices, objective_keys)
        init_action = self._merge_rerank_warm_start(init_action, rerank_init_action, need_high_replan, clean_info)
        outputs = self._solve_low_level(clean_info, init_action, use_short_refinement=rerank_init_action is not None and self.subgoal_rerank_reuse_actions)
        self.model._cost_mode = 'default'
        self._advance_subgoal_progress(env_indices)
        self._update_histories(env_indices, cur_z_l, z_h_current)
        outputs['solve_time'] = time.time() - start_time
        return outputs

    def _strip_metadata(self, info_dict: dict) -> dict:
        out = {k: v for (k, v) in info_dict.items()}
        out.pop('_env_indices', None)
        out.pop('_needs_flush', None)
        return out

    def _final_goal_solve_with_cache(self, clean_info: dict, init_action: torch.Tensor | None, env_indices: torch.Tensor) -> dict:
        outputs: dict = {'costs': []}
        if self.final_receding_horizon <= 1:
            outputs = self.solver_a(clean_info, init_action=init_action)
            return outputs
        idxs = env_indices.detach().cpu().tolist()
        solve_rows: list[int] = []
        cached_rows: list[int] = []
        for (row, env_i) in enumerate(idxs):
            plan = self._final_plan_cache[env_i]
            if plan is not None and self._final_plan_consumed[env_i] < self.final_receding_horizon:
                cached_rows.append(row)
            else:
                solve_rows.append(row)
        actions_parts: list[tuple[int, torch.Tensor]] = []
        if solve_rows:
            rows_t = torch.as_tensor(solve_rows, device=env_indices.device, dtype=torch.long)
            sub_info = self._slice_info(clean_info, rows_t)
            sub_init = init_action[rows_t] if init_action is not None else None
            sub_outputs = self.solver_a(sub_info, init_action=sub_init)
            sub_actions = sub_outputs['actions']
            sub_costs = list(sub_outputs.get('costs', []))
            for (j, row) in enumerate(solve_rows):
                env_i = idxs[row]
                self._final_plan_cache[env_i] = sub_actions[j]
                self._final_plan_consumed[env_i] = 1
                actions_parts.append((row, sub_actions[j]))
                cost = sub_costs[j] if j < len(sub_costs) else float('nan')
                outputs['costs'].append(cost)
        for row in cached_rows:
            env_i = idxs[row]
            c = self._final_plan_consumed[env_i]
            plan = self._final_plan_cache[env_i]
            pad_value = -1 if self._discrete_low_level else 0
            padding = plan.new_full((c, plan.shape[-1]), pad_value)
            shifted = torch.cat([plan[c:], padding], dim=0)
            self._final_plan_consumed[env_i] = c + 1
            actions_parts.append((row, shifted))
            outputs['costs'].append(float('nan'))
        actions_parts.sort(key=lambda t: t[0])
        outputs['actions'] = torch.stack([a for (_, a) in actions_parts], dim=0)
        return outputs

    def _solve_split_by_final_mask(self, info_dict: dict, init_action: torch.Tensor | None, final_mask: torch.Tensor, env_indices: torch.Tensor, cur_z_l: torch.Tensor, z_h_current: torch.Tensor, start_time: float | None=None) -> dict:
        if start_time is None:
            start_time = time.time()
        non_final_idx = torch.where(~final_mask)[0]
        final_idx = torch.where(final_mask)[0]
        outputs = {'costs': [], 'actions': None}
        if len(non_final_idx) > 0:
            nf_info = self._slice_info(info_dict, non_final_idx)
            nf_init = init_action[non_final_idx] if init_action is not None else None
            nf_env_indices = env_indices[non_final_idx]
            need_high_replan = self._activate_cached_or_request_replan(nf_env_indices)
            rerank_init_action = None
            if any(need_high_replan):
                z_h_ctx = self._build_high_level_predictor_context(z_h_current[non_final_idx], nf_env_indices)
                nf_info['z_h_context'] = z_h_ctx
                clean_info = self._strip_metadata(nf_info)
                (z_h_plan, u_outputs, rerank_init_action) = self._plan_high_level(clean_info, z_h_ctx)
                self._assign_planned_high_level_subgoals(nf_env_indices, need_high_replan, z_h_plan, u_outputs)
            z_h_subgoal = [self._current_z_h_subgoal[env_i] for env_i in nf_env_indices.tolist()]
            nf_info['z_h_subgoal'] = torch.stack(z_h_subgoal, dim=0)
            (prefix, prefix_len) = self._build_subgoal_prefix(nf_env_indices, cur_z_l[non_final_idx])
            nf_info['z_l_subgoal_prefix'] = prefix
            nf_info['z_l_subgoal_prefix_len'] = prefix_len
            self.model._cost_mode = 'low_level_subgoal'
            clean_info = self._strip_metadata(nf_info)
            objective_type = 'high_subgoal'
            objective_keys = [(objective_type, self._high_level_subgoal_assignment_count[env_i]) for env_i in nf_env_indices.detach().cpu().tolist()]
            (nf_init, warm_start_reset) = self._prepare_low_level_warm_start(clean_info, nf_init, nf_env_indices, objective_keys)
            nf_init = self._merge_rerank_warm_start(nf_init, rerank_init_action, need_high_replan, clean_info)
            nf_outputs = self._solve_low_level(clean_info, nf_init, use_short_refinement=rerank_init_action is not None and self.subgoal_rerank_reuse_actions)
            self.model._cost_mode = 'default'
            self._advance_subgoal_progress(nf_env_indices)
            self._update_histories(nf_env_indices, cur_z_l[non_final_idx], z_h_current[non_final_idx])
            outputs['costs'].extend(nf_outputs.get('costs', []))
            nf_actions = nf_outputs['actions']
        if len(final_idx) > 0:
            f_info = self._slice_info(info_dict, final_idx)
            f_init = init_action[final_idx] if init_action is not None else None
            f_env_indices = env_indices[final_idx]
            self.model._cost_mode = 'final_goal'
            clean_info = self._strip_metadata(f_info)
            (f_init, warm_start_reset) = self._prepare_low_level_warm_start(clean_info, f_init, f_env_indices, [('final_goal', 0)] * len(f_env_indices))
            f_outputs = self._final_goal_solve_with_cache(clean_info, f_init, f_env_indices)
            self.model._cost_mode = 'default'
            self._update_histories(f_env_indices, cur_z_l[final_idx], z_h_current[final_idx])
            outputs['costs'].extend(f_outputs.get('costs', []))
            f_actions = f_outputs['actions']
        all_actions = torch.zeros(len(env_indices), self.horizon, self._low_action_dim, device='cpu', dtype=torch.long if self._discrete_low_level else torch.float32)
        if len(non_final_idx) > 0:
            all_actions[non_final_idx] = nf_actions.cpu()
        if len(final_idx) > 0:
            all_actions[final_idx] = f_actions.cpu()
        outputs['actions'] = all_actions
        outputs['solve_time'] = time.time() - start_time
        return outputs

    def _slice_info(self, info_dict: dict, idx: torch.Tensor) -> dict:
        out = {}
        for (k, v) in info_dict.items():
            if torch.is_tensor(v):
                out[k] = v[idx]
            elif isinstance(v, np.ndarray):
                out[k] = v[idx.cpu().numpy()]
            elif isinstance(v, list):
                out[k] = [v[i] for i in idx.cpu().tolist()]
            else:
                out[k] = v
        if '_env_indices' in info_dict:
            out['_env_indices'] = info_dict['_env_indices'][idx]
        return out
