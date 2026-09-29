import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_worldmodel.wm.lewm.module import Block, Transformer

class BidirectionalBlock(Block):

    def forward(self, x):
        x = x + self.attn(self.norm1(x), causal=False)
        x = x + self.mlp(self.norm2(x))
        return x

class TemporalEncoder(nn.Module):

    def __init__(self, *, input_dim, hidden_dim=None, output_dim=None, depth=4, heads=8, dim_head=64, mlp_dim=None, dropout=0.1, pooling='mean'):
        super().__init__()
        if pooling not in ('mean', 'last_token'):
            raise ValueError(f"pooling must be 'mean' or 'last_token', got {pooling!r}")
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        mlp_dim = mlp_dim or 4 * hidden_dim
        self.pooling = pooling
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, input_dim))
        self.transformer = Transformer(input_dim, hidden_dim, hidden_dim, depth, heads, dim_head, mlp_dim, dropout, block_class=BidirectionalBlock)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.transformer(x)
        if self.pooling == 'last_token':
            x = x[:, -1]
        else:
            x = x.mean(dim=1)
        x = self.head(x)
        return x

class MacroActionEncoder(nn.Module):

    def __init__(self, *, input_dim, output_dim, u_dim=16, hidden_dim=None, depth=4, heads=8, dim_head=64, mlp_dim=None, dropout=0.1, latent_mode='sigreg', logvar_min=-10.0, logvar_max=5.0):
        super().__init__()
        if latent_mode not in ('sigreg', 'variational'):
            raise ValueError(f'MacroActionEncoder latent_mode must be sigreg|variational, got {latent_mode!r}')
        if logvar_min >= logvar_max:
            raise ValueError('MacroActionEncoder requires logvar_min < logvar_max')
        hidden_dim = hidden_dim or output_dim
        mlp_dim = mlp_dim or 4 * hidden_dim
        self.u_dim = u_dim
        self.latent_mode = latent_mode
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, hidden_dim))
        self.transformer = Transformer(hidden_dim, hidden_dim, hidden_dim, depth, heads, dim_head, mlp_dim, dropout, block_class=BidirectionalBlock)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_u = nn.Linear(hidden_dim, u_dim)
        if latent_mode == 'variational':
            self.to_logvar = nn.Linear(hidden_dim, u_dim)
        self.from_u = nn.Linear(u_dim, output_dim)

    def _features(self, x):
        T = x.size(1)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :T]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.norm(x)

    def forward(self, x, return_u=False, return_stats=False, sample=None):
        x = self._features(x)
        mean = self.to_u(x)
        stats = {'mean': mean}
        if self.latent_mode == 'variational':
            logvar = self.to_logvar(x).clamp(min=self.logvar_min, max=self.logvar_max)
            stats['logvar'] = logvar
            should_sample = self.training if sample is None else sample
            if should_sample:
                u = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            else:
                u = mean
        else:
            u = mean
        out = self.from_u(u)
        if return_stats:
            if not return_u:
                raise ValueError('return_stats=True requires return_u=True')
            return (out, u, stats)
        if return_u:
            return (out, u)
        return out

class _LogitScaleMixin:

    def _init_logit_scale(self, *, temperature, learnable_temperature=True, max_logit_scale=None):
        init = torch.ones([]) * math.log(1 / temperature)
        if learnable_temperature:
            self.logit_scale = nn.Parameter(init)
        else:
            self.register_buffer('logit_scale', init)
        self.max_logit_scale = max_logit_scale

    def _scale_scores(self, scores):
        scale = self.logit_scale.exp()
        if self.max_logit_scale is not None:
            scale = scale.clamp(max=self.max_logit_scale)
        scores = scores * scale
        bias = getattr(self, 'logit_bias', None)
        if bias is not None:
            scores = scores + bias
        return scores

    def _init_logit_bias(self, bias_init=None, learnable_bias=True):
        self.bias_init = bias_init
        if bias_init is None:
            self.register_parameter('logit_bias', None)
            self.register_buffer('_logit_bias_initialized', torch.tensor(True), persistent=False)
            return
        if isinstance(bias_init, str):
            if bias_init != 'label_prior':
                raise ValueError(f"bias_init must be None, 'label_prior', or a number, got {bias_init!r}")
            initial_value = 0.0
            initialized = False
        else:
            initial_value = float(bias_init)
            initialized = True
        value = torch.tensor(initial_value, dtype=torch.float32)
        if learnable_bias:
            self.logit_bias = nn.Parameter(value)
        else:
            self.register_buffer('logit_bias', value)
        self.register_buffer('_logit_bias_initialized', torch.tensor(initialized, dtype=torch.bool))

    @property
    def logit_bias_initialized(self):
        return bool(self._logit_bias_initialized.item())

    @torch.no_grad()
    def initialize_logit_bias_from_targets(self, targets, eps=1e-06):
        if self.bias_init != 'label_prior' or self.logit_bias_initialized:
            return False
        mean_target = targets.detach().float().mean().clamp(eps, 1.0 - eps)
        self.logit_bias.copy_(torch.logit(mean_target).to(self.logit_bias))
        self._logit_bias_initialized.fill_(True)
        return True

class CrossLevelCompatibility(nn.Module, _LogitScaleMixin):

    def __init__(self, *, dim, temperature=0.07, learnable_temperature=True, max_logit_scale=None, bias_init=None, learnable_bias=True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self._init_logit_scale(temperature=temperature, learnable_temperature=learnable_temperature, max_logit_scale=max_logit_scale)
        self._init_logit_bias(bias_init, learnable_bias)

    def _project(self, z_h, z_l):
        return (F.normalize(z_h @ self.W, dim=-1), F.normalize(z_l, dim=-1))

    def forward(self, z_h, z_l):
        (z_h_proj, z_l_norm) = self._project(z_h, z_l)
        scores = z_h_proj @ z_l_norm.T
        return self._scale_scores(scores)

    def score_pairs(self, z_h, z_l):
        if z_h.shape != z_l.shape:
            raise ValueError(f'aligned pair shapes must match, got {z_h.shape} and {z_l.shape}')
        (z_h_proj, z_l_proj) = self._project(z_h, z_l)
        return self._scale_scores((z_h_proj * z_l_proj).sum(dim=-1))

class CrossLevelCompatibilityMLP(nn.Module, _LogitScaleMixin):

    def __init__(self, *, dim, hidden_dim=None, proj_dim=None, depth=2, dropout=0.0, temperature=0.07, learnable_temperature=True, max_logit_scale=None, bias_init=None, learnable_bias=True):
        super().__init__()
        hidden_dim = hidden_dim or 2 * dim
        proj_dim = proj_dim or dim
        self.z_h_proj = self._build_tower(dim, hidden_dim, proj_dim, depth, dropout)
        self.z_l_proj = self._build_tower(dim, hidden_dim, proj_dim, depth, dropout)
        self._init_logit_scale(temperature=temperature, learnable_temperature=learnable_temperature, max_logit_scale=max_logit_scale)
        self._init_logit_bias(bias_init, learnable_bias)

    def _project(self, z_h, z_l):
        return (F.normalize(self.z_h_proj(z_h), dim=-1), F.normalize(self.z_l_proj(z_l), dim=-1))

    @staticmethod
    def _build_tower(input_dim, hidden_dim, output_dim, depth, dropout):
        if depth <= 1:
            return nn.Linear(input_dim, output_dim)
        layers = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.extend([nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.extend([nn.LayerNorm(in_dim), nn.Linear(in_dim, output_dim)])
        return nn.Sequential(*layers)

    def forward(self, z_h, z_l):
        (z_h_proj, z_l_proj) = self._project(z_h, z_l)
        scores = z_h_proj @ z_l_proj.T
        return self._scale_scores(scores)

    def score_pairs(self, z_h, z_l):
        if z_h.shape != z_l.shape:
            raise ValueError(f'aligned pair shapes must match, got {z_h.shape} and {z_l.shape}')
        (z_h_proj, z_l_proj) = self._project(z_h, z_l)
        return self._scale_scores((z_h_proj * z_l_proj).sum(dim=-1))
__all__ = ['BidirectionalBlock', 'TemporalEncoder', 'MacroActionEncoder', 'CrossLevelCompatibility', 'CrossLevelCompatibilityMLP']
