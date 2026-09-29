import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from stable_worldmodel.wm.planning_cost import future_predictions, goal_planning_cost

class DualWM(nn.Module):

    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None, temporal_encoder=None, high_level_predictor=None, macro_action_encoder=None, compatibility=None, window_size=4, high_level_history_size=2, high_level_predictor_context='history', macro_action_alignment='endpoint_transition', **kwargs):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.temporal_encoder = temporal_encoder
        self.high_level_predictor = high_level_predictor
        self.macro_action_encoder = macro_action_encoder
        self.compatibility = compatibility
        self.window_size = window_size
        self.high_level_history_size = high_level_history_size
        if high_level_predictor_context not in ('history', 'current_only'):
            raise ValueError(f'high_level_predictor_context must be history|current_only, got {high_level_predictor_context!r}')
        if high_level_predictor_context == 'current_only' and high_level_history_size != 1:
            raise ValueError('current_only high-level predictor requires high_level_history_size=1')
        predictor_num_frames = getattr(high_level_predictor, 'num_frames', high_level_history_size)
        if predictor_num_frames != high_level_history_size:
            raise ValueError(f'high_level_predictor.num_frames must match high_level_history_size, got {predictor_num_frames} and {high_level_history_size}')
        self.high_level_predictor_context = high_level_predictor_context
        self.macro_action_alignment = macro_action_alignment
        self._cost_mode = 'default'
        self.high_level_goal_cost_mode = 'compatibility'
        self.subgoal_cost_mode = 'dynamic_window_aligned'

    def encode(self, info, return_visual_aux=False):
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        emb = self.projector(output.last_hidden_state[:, 0])
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)
        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info

    def predict(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    def low_level_rollout_train(self, emb, act_emb, history_size=3, num_rollout_steps=5):
        (B, T, D) = emb.shape
        HS = history_size
        ctx_emb = emb[:, :HS]
        ctx_act = act_emb[:, :HS]
        pred_list = []
        tgt_list = []
        for k_step in range(num_rollout_steps):
            pred = self.predict(ctx_emb, ctx_act)
            pred_list.append(pred)
            tgt_list.append(emb[:, k_step + 1:k_step + 1 + HS])
            ctx_emb = torch.cat([ctx_emb[:, 1:], pred[:, -1:]], dim=1)
            ctx_act = torch.cat([ctx_act[:, 1:], act_emb[:, HS + k_step:HS + k_step + 1]], dim=1)
        return (pred_list, tgt_list)

    def rollout(self, info, action_sequence, history_size=None):
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)
        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        (B, S, T) = action_sequence.shape[:3]
        (act_0, act_future) = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H
        if 'emb' not in info:
            _init = {k: v[:, 0] for (k, v) in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_flat = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future_flat = rearrange(act_future, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(torch.cat([act_flat, act_future_flat], dim=1))
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo:H + t]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])
        emb = torch.stack(emb_list, dim=1)
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout
        return info

    def criterion(self, info_dict: dict):
        return goal_planning_cost(self, info_dict)

    def encode_high_level(self, emb, k=None):
        k = k or self.window_size
        (B, T, D) = emb.shape
        assert T % k == 0, f'T={T} not divisible by k={k}'
        T_h = T // k
        emb_windows = emb.view(B, T_h, k, D).reshape(B * T_h, k, D)
        z_h = self.temporal_encoder(emb_windows)
        return z_h.reshape(B, T_h, -1)

    def encode_macro_action(self, action, k=None, return_u=False, alignment=None, return_stats=False, sample=None):
        k = k or self.window_size
        alignment = alignment or self.macro_action_alignment
        if return_stats and (not return_u):
            raise ValueError('return_stats=True requires return_u=True')
        (B, T, A_frame) = action.shape
        assert T % k == 0, f'T={T} not divisible by k={k}'
        T_h = T // k
        if alignment in (None, 'next_window', 'window'):
            action_windows = action.view(B, T_h, k, A_frame).reshape(B * T_h, k, A_frame)
            kwargs = {'return_u': True}
            if return_stats:
                kwargs['return_stats'] = True
            if sample is not None:
                kwargs['sample'] = sample
            if return_u:
                encoded = self.macro_action_encoder(action_windows, **kwargs)
                if return_stats:
                    (out, u, stats) = encoded
                    stats = {name: value.reshape(B, T_h, -1) for (name, value) in stats.items()}
                    return (out.reshape(B, T_h, -1), u.reshape(B, T_h, -1), stats)
                (out, u) = encoded
                return (out.reshape(B, T_h, -1), u.reshape(B, T_h, -1))
            out = self.macro_action_encoder(action_windows)
            return out.reshape(B, T_h, -1)
        if alignment == 'endpoint_transition':
            if T_h < 2:
                raise ValueError(f'endpoint_transition requires at least two high-level windows, got T_h={T_h}')
            starts = [i * k - 1 for i in range(1, T_h)]
            action_windows = torch.stack([action[:, start:start + k] for start in starts], dim=1).reshape(B * (T_h - 1), k, A_frame)
            if return_u:
                kwargs = {'return_u': True}
                if return_stats:
                    kwargs['return_stats'] = True
                if sample is not None:
                    kwargs['sample'] = sample
                encoded = self.macro_action_encoder(action_windows, **kwargs)
                if return_stats:
                    (out_valid, u_valid, stats_valid) = encoded
                else:
                    (out_valid, u_valid) = encoded
                out_dim = out_valid.shape[-1]
                u_dim = u_valid.shape[-1]
                out = out_valid.new_zeros(B, T_h, out_dim)
                u = u_valid.new_zeros(B, T_h, u_dim)
                out[:, 1:] = out_valid.reshape(B, T_h - 1, out_dim)
                u[:, 1:] = u_valid.reshape(B, T_h - 1, u_dim)
                if return_stats:
                    stats = {}
                    for (name, value_valid) in stats_valid.items():
                        value = value_valid.new_zeros(B, T_h, u_dim)
                        value[:, 1:] = value_valid.reshape(B, T_h - 1, u_dim)
                        stats[name] = value
                    return (out, u, stats)
                return (out, u)
            out_valid = self.macro_action_encoder(action_windows)
            out_dim = out_valid.shape[-1]
            out = out_valid.new_zeros(B, T_h, out_dim)
            out[:, 1:] = out_valid.reshape(B, T_h - 1, out_dim)
            return out
        raise ValueError(f'Unknown macro_action_alignment={alignment!r}; expected next_window|endpoint_transition')

    def high_level_rollout_train_from_latents(self, z_h_gt, u, num_rollout_steps=1):
        HS_H = self.high_level_history_size
        T_h = z_h_gt.shape[1]
        assert T_h == u.shape[1], f'z_h_gt T_h={T_h} must match u T_h={u.shape[1]}'
        assert HS_H + num_rollout_steps <= T_h, f'HS_H + num_rollout_steps={HS_H + num_rollout_steps} > T_h={T_h}'
        ctx_z_h = z_h_gt[:, :HS_H]
        ctx_u = u[:, 1:HS_H + 1]
        pred_list = []
        tgt_list = []
        for m in range(num_rollout_steps):
            pred = self.high_level_predictor(ctx_z_h, ctx_u)
            z_h_pred = pred[:, -1]
            pred_list.append(z_h_pred)
            tgt_list.append(z_h_gt[:, HS_H + m])
            ctx_z_h = torch.cat([ctx_z_h[:, 1:], z_h_pred.unsqueeze(1)], dim=1)
            if m < num_rollout_steps - 1:
                ctx_u = torch.cat([ctx_u[:, 1:], u[:, HS_H + m + 1:HS_H + m + 2]], dim=1)
        return (pred_list, tgt_list)

    def high_level_rollout_train(self, emb, action, k=None, num_rollout_steps=1, alignment=None):
        k = k or self.window_size
        z_h_gt = self.encode_high_level(emb, k)
        u = self.encode_macro_action(action, k, alignment=alignment)
        return self.high_level_rollout_train_from_latents(z_h_gt, u, num_rollout_steps=num_rollout_steps)

    def get_cost(self, info_dict, action_candidates):
        mode = getattr(self, '_cost_mode', 'default')
        if mode == 'high_level':
            return self._get_cost_high_level(info_dict, action_candidates)
        elif mode == 'low_level_subgoal':
            return self._get_cost_low_level_subgoal(info_dict, action_candidates)
        elif mode == 'final_goal':
            return self._get_cost_final_goal(info_dict, action_candidates)
        return self._get_cost_default(info_dict, action_candidates)

    def _get_cost_default(self, info_dict, action_candidates):
        assert 'goal' in info_dict, 'goal not in info_dict'
        if 'goal_emb' not in info_dict:
            goal = {k: v[:, 0] for (k, v) in info_dict.items() if torch.is_tensor(v)}
            goal['pixels'] = goal['goal']
            for k in info_dict:
                if k.startswith('goal_'):
                    goal[k[len('goal_'):]] = goal.pop(k)
            goal.pop('action')
            goal = self.encode(goal)
            info_dict['goal_emb'] = goal['emb']
        info_dict = self.rollout(info_dict, action_candidates)
        cost = self.criterion(info_dict)
        return cost

    def rollout_high_level_from_context(self, z_h_context, u_candidates):
        has_samples = u_candidates.ndim == 4
        if not has_samples:
            u_candidates = u_candidates[:, None]
        (B, S, H_u, _) = u_candidates.shape
        HS_H = self.high_level_history_size
        D = z_h_context.shape[-1]
        z_ctx = z_h_context[:, None].expand(B, S, HS_H, D)
        u_high = self.macro_action_encoder.from_u(u_candidates)
        zero_u = u_high.new_zeros(B, S, 1, D)
        u_hist = zero_u.expand(B, S, max(1, HS_H - 1), D)
        z_seq = [z_ctx[:, :, i] for i in range(HS_H)]
        u_seq = [u_hist[:, :, i] for i in range(HS_H - 1)]
        preds = []
        for t in range(H_u):
            u_seq.append(u_high[:, :, t])
            z_in = torch.stack(z_seq[-HS_H:], dim=2)
            u_in = torch.stack(u_seq[-HS_H:], dim=2)
            z_next = self.high_level_predictor(z_in.reshape(B * S, HS_H, D), u_in.reshape(B * S, HS_H, D))[:, -1].reshape(B, S, D)
            preds.append(z_next[:, 0] if not has_samples else z_next)
            z_seq.append(z_next)
        return preds

    def _get_cost_high_level(self, info_dict, macro_action_candidates):
        if self.high_level_goal_cost_mode != 'lift_mse':
            raise ValueError('The released planner uses high_level_goal_cost.mode=lift_mse')
        z_h_context = info_dict['z_h_context']
        (B, S, HS_H, D) = z_h_context.shape
        z_h_final = self.rollout_high_level_from_context(z_h_context[:, 0], macro_action_candidates)[-1]
        z_g = info_dict['goal_emb']
        k = self.window_size
        z_g_h = self.temporal_encoder(z_g.unsqueeze(2).expand(B, S, k, D).reshape(B * S, k, D)).reshape(B, S, D)
        return F.mse_loss(z_h_final, z_g_h.detach(), reduction='none').sum(dim=-1)

    def _get_cost_low_level_subgoal(self, info_dict, action_candidates):
        if self.subgoal_cost_mode != 'dynamic_window_aligned':
            raise ValueError('The released planner uses subgoal_cost.mode=dynamic_window_aligned')
        info_dict = self.rollout(info_dict, action_candidates)
        predicted_only = future_predictions(info_dict)
        dists = self._subgoal_aligned_window_distances(predicted_only, info_dict['z_h_subgoal'], info_dict.get('z_l_subgoal_prefix'), info_dict.get('z_l_subgoal_prefix_len'))
        return dists[:, :, 0]

    def _subgoal_aligned_window_distances(self, predicted_only, z_h_subgoal, prefix=None, prefix_len=None):
        (B, S, T, D) = predicted_only.shape
        k = self.window_size
        if prefix is None:
            prefix = predicted_only.new_zeros(B, S, k - 1, D)
        elif prefix.ndim == 3:
            prefix = prefix[:, None].expand(B, S, -1, -1)
        if prefix.shape != (B, S, k - 1, D):
            raise ValueError(f'z_l_subgoal_prefix must broadcast to (B,S,k-1,D), got {tuple(prefix.shape)} vs {(B, S, k - 1, D)}')
        if prefix_len is None:
            prefix_len = torch.zeros(B, S, device=predicted_only.device)
        elif prefix_len.ndim == 1:
            prefix_len = prefix_len[:, None].expand(B, S)
        if prefix_len.shape != (B, S):
            raise ValueError(f'z_l_subgoal_prefix_len must broadcast to (B,S), got {tuple(prefix_len.shape)} vs {(B, S)}')
        prefix_len = prefix_len.to(device=predicted_only.device, dtype=torch.long)
        per_env_len = prefix_len[:, :1]
        if not torch.equal(prefix_len, per_env_len.expand_as(prefix_len)):
            raise ValueError('All CEM samples for one env need one prefix length')
        if bool(((per_env_len < 0) | (per_env_len >= k)).any()):
            raise ValueError(f'z_l_subgoal_prefix_len must be in [0,{k - 1}]')
        if T == 0 or bool((k - per_env_len > T).any()):
            raise ValueError(f'predicted_only T={T} is too short for the remaining dynamic_window_aligned suffix')
        positions = torch.arange(k, device=predicted_only.device).view(1, 1, k)
        observed_mask = positions < prefix_len.unsqueeze(-1)
        predicted_indices = (positions - prefix_len.unsqueeze(-1)).clamp(min=0, max=T - 1)
        predicted_window = torch.gather(predicted_only, dim=2, index=predicted_indices.unsqueeze(-1).expand(B, S, k, D))
        prefix_window = torch.cat([prefix, prefix.new_zeros(B, S, 1, D)], dim=2)
        z_l_window = torch.where(observed_mask.unsqueeze(-1), prefix_window, predicted_window)
        z_h_pred = self.temporal_encoder(z_l_window.reshape(B * S, k, D)).reshape(B, S, D)
        if z_h_subgoal.ndim == 2:
            z_h_subgoal = z_h_subgoal[:, None, :]
        if z_h_subgoal.shape[1] == 1 and S != 1:
            z_h_subgoal = z_h_subgoal.expand(B, S, D)
        if z_h_subgoal.shape != (B, S, D):
            raise ValueError(f'z_h_subgoal must broadcast to (B,S,D), got {tuple(z_h_subgoal.shape)} vs {(B, S, D)}')
        target = z_h_subgoal.detach().expand_as(z_h_pred)
        return F.mse_loss(z_h_pred, target, reduction='none').sum(dim=-1, keepdim=True)

    def _get_cost_final_goal(self, info_dict, action_candidates):
        return self._get_cost_default(info_dict, action_candidates)
__all__ = ['DualWM']
