import json
import math
import re
from pathlib import Path
import torch
from loguru import logger as logging
import torch.nn.functional as F
from stable_worldmodel.data.utils import get_cache_dir, ensure_dir_exists
_VIT_LAYER_SUFFIX_PAIRS = (('attention.q_proj.', 'attention.attention.query.'), ('attention.k_proj.', 'attention.attention.key.'), ('attention.v_proj.', 'attention.attention.value.'), ('attention.o_proj.', 'attention.output.dense.'), ('mlp.fc1.', 'intermediate.dense.'), ('mlp.fc2.', 'output.dense.'), ('layernorm_before.', 'layernorm_before.'), ('layernorm_after.', 'layernorm_after.'))
_BARE_VIT_LAYER_PATTERN = re.compile('^(?P<prefix>.*(?:^|\\.)encoder)\\.layers\\.(?P<index>\\d+)\\.(?P<suffix>.+)$')
_WRAPPED_VIT_LAYER_PATTERN = re.compile('^(?P<prefix>.*(?:^|\\.)encoder)\\.encoder\\.layer\\.(?P<index>\\d+)\\.(?P<suffix>.+)$')

def _vit_encoder_key_counterpart(key: str) -> str | None:
    match = _BARE_VIT_LAYER_PATTERN.match(key)
    suffix_pairs = _VIT_LAYER_SUFFIX_PAIRS
    if match is None:
        match = _WRAPPED_VIT_LAYER_PATTERN.match(key)
        suffix_pairs = tuple(((wrapped, bare) for (bare, wrapped) in _VIT_LAYER_SUFFIX_PAIRS))
    if match is None:
        return None
    suffix = match.group('suffix')
    for (source_suffix, target_suffix) in suffix_pairs:
        if suffix.startswith(source_suffix):
            middle = 'encoder.layer' if match.re is _BARE_VIT_LAYER_PATTERN else 'layers'
            return f"{match.group('prefix')}.{middle}.{match.group('index')}.{target_suffix}{suffix[len(source_suffix):]}"
    return None

def remap_checkpoint_encoder_keys(state_dict: dict[str, torch.Tensor], target_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    remapped = {}
    sources = {}
    for (source_key, value) in state_dict.items():
        target_key = source_key
        if source_key not in target_state:
            candidate = _vit_encoder_key_counterpart(source_key)
            if candidate in target_state:
                target_key = candidate
        if target_key in remapped:
            raise ValueError(f'Checkpoint key remapping collision: {source_key!r} and {sources[target_key]!r} -> {target_key!r}')
        if target_key != source_key:
            target_shape = target_state[target_key].shape
            if value.shape != target_shape:
                raise ValueError(f'Checkpoint shape mismatch after encoder-key remap: {source_key!r} {tuple(value.shape)} -> {target_key!r} {tuple(target_shape)}')
        remapped[target_key] = value
        sources[target_key] = source_key
    renamed_count = sum((key != source for (key, source) in sources.items()))
    return (remapped, renamed_count)

def save_pretrained(model: torch.nn.Module, run_name: str, config: dict | None=None, config_key: str | None=None, filename: str='weights.pt', cache_dir: str=None):
    from omegaconf import OmegaConf
    ckpt_dir = get_cache_dir(cache_dir, sub_folder='checkpoints') / run_name
    ensure_dir_exists(ckpt_dir)
    checkpoint_path = ckpt_dir / filename
    torch.save(model.state_dict(), checkpoint_path)
    if config is None:
        logging.warning('No config! Loading will have to be done manually.')
        return
    if config_key is not None and config_key in config:
        config = config[config_key]
    config_path = ckpt_dir / 'config.json'
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logging.info(f'📦📦📦 Model saved to {checkpoint_path} 📦📦📦')
    return
__all__ = ['load_pretrained', 'save_pretrained']

def load_pretrained(name, cache_dir=None, extra_args=None):
    from hydra.utils import instantiate
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = get_cache_dir(cache_dir, 'checkpoints') / path
    if path.is_dir():
        named_checkpoint = path / f'{path.name}.pt'
        path = named_checkpoint if named_checkpoint.is_file() else path / 'weights.pt'
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads((path.parent / 'config.json').read_text())
    config = config.get('model', config)
    if extra_args:
        from omegaconf import OmegaConf
        config = OmegaConf.merge(config, OmegaConf.from_dotlist([f'{key}={value}' for (key, value) in extra_args.items()]))
    model = instantiate(config)
    state = torch.load(path, map_location='cpu', weights_only=True)
    (state, _) = remap_checkpoint_encoder_keys(state, model.state_dict())
    model.load_state_dict(state, strict=True)
    return model

def goal_distance(model, pred_future: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    metric = str(getattr(model, 'planning_cost_distance', None) or 'euclidean')
    goal = goal.detach().expand_as(pred_future)
    if metric == 'euclidean':
        return F.mse_loss(pred_future, goal, reduction='none').sum(dim=-1)
    if metric == 'cosine':
        return 1.0 - F.cosine_similarity(pred_future, goal, dim=-1)
    if metric == 'geodesic':
        eps = 1e-06
        z1 = F.normalize(pred_future, dim=-1)
        z2 = F.normalize(goal, dim=-1)
        cos = (z1 * z2).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
        return torch.arccos(cos)
    raise ValueError(f'Unknown planning_cost_distance={metric!r}; expected euclidean|cosine|geodesic')


def MSE_Cost(model, pred_future: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    dist = goal_distance(model, pred_future, goal)
    reach = -0.1 * (torch.logsumexp(-dist / 0.1, dim=-1) - math.log(dist.shape[-1]))
    trajectory = dist.mean(dim=-1)
    return reach + 0.01 * trajectory
