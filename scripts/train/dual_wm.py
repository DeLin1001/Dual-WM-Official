import json
import math
import os
from pathlib import Path
from functools import partial
import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf, open_dict
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.data.split import split_by_episode
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.dual_wm.loss import diagonal_gaussian_kl
from stable_worldmodel.wm.utils import remap_checkpoint_encoder_keys, save_pretrained
LOW_LEVEL_MODULE_PREFIXES = ('encoder.', 'predictor.', 'action_encoder.', 'projector.', 'pred_proj.')

def get_img_preprocessor(source: str, target: str, img_size: int=224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)

def resolve_checkpoint_path(path_like, cache_dir=None):
    path = Path(path_like)
    if path.is_absolute() or path.exists():
        return path
    return Path(swm.data.utils.get_cache_dir(cache_dir, sub_folder='checkpoints'), path)

def validate_low_only_checkpoint_keys(state_dict, target_state):
    expected = {key for key in target_state if key.startswith(LOW_LEVEL_MODULE_PREFIXES)}
    actual = set(state_dict)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f'low_only checkpoint does not exactly cover the frozen low-level modules: missing={missing[:10]}, unexpected={unexpected[:10]}')
    return len(expected)

class SaveCkptCallback(Callback):

    def __init__(self, run_name, cfg, epoch_interval: int=1, cache_dir=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.cache_dir = cache_dir

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)
            if trainer.current_epoch + 1 == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(model, run_name=self.run_name, config=self.cfg, filename=f'weights_epoch_{epoch}.pt', cache_dir=self.cache_dir)

    def on_train_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            save_pretrained(pl_module.model, run_name=self.run_name, config=self.cfg, filename='weights_final.pt', cache_dir=self.cache_dir)

def weighted_mean(losses, weighting='exponential', alpha=0.5):
    if not losses:
        raise ValueError('Rollout losses must be nonempty')
    if weighting != 'exponential' or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Use exponential rollout weighting with a finite positive alpha')
    raw = [math.exp(-alpha * k) for k in range(len(losses))]
    total = sum(raw)
    return sum((weight / total * loss for weight, loss in zip(raw, losses)))

def apply_train_control(world_model, cfg):
    tc = cfg.get('train_control', {})
    freeze_names = list(tc.get('freeze_modules', []))
    eval_frozen = tc.get('eval_frozen_modules', True)
    for name in freeze_names:
        module = getattr(world_model, name, None)
        if module is None:
            print(f"  [train_control] WARNING: module '{name}' not found, skipping")
            continue
        for param in module.parameters():
            param.requires_grad_(False)
        if eval_frozen:
            module.eval()
        print(f'  [train_control] frozen: {name} (eval={eval_frozen})')
    return freeze_names

class KeepFrozenEvalCallback(Callback):

    def __init__(self, frozen_module_names):
        super().__init__()
        self.frozen_module_names = frozen_module_names
        self._reference_state = None

    def _reapply_eval(self, pl_module):
        for name in self.frozen_module_names:
            module = getattr(pl_module.model, name, None)
            if module is not None:
                module.eval()

    def on_train_start(self, trainer, pl_module):
        self._reapply_eval(pl_module)
        self._reference_state = {f'{module_name}.{state_name}': tensor.detach().cpu().clone() for module_name in self.frozen_module_names for module in [getattr(pl_module.model, module_name, None)] if module is not None for (state_name, tensor) in module.state_dict().items()}

    def on_train_epoch_start(self, trainer, pl_module):
        self._reapply_eval(pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._reapply_eval(pl_module)

    def _assert_unchanged(self, pl_module):
        if self._reference_state is None:
            return
        changed = []
        for module_name in self.frozen_module_names:
            module = getattr(pl_module.model, module_name, None)
            if module is None:
                continue
            for (state_name, tensor) in module.state_dict().items():
                key = f'{module_name}.{state_name}'
                reference = self._reference_state.get(key)
                if reference is None or not torch.equal(tensor.detach().cpu(), reference):
                    changed.append(key)
        if changed:
            preview = ', '.join(changed[:8])
            suffix = '' if len(changed) <= 8 else f' (+{len(changed) - 8} more)'
            raise RuntimeError(f'Frozen module state changed during training: {preview}{suffix}')

    def on_train_epoch_end(self, trainer, pl_module):
        self._assert_unchanged(pl_module)

    def on_train_end(self, trainer, pl_module):
        self._assert_unchanged(pl_module)

def dual_wm_dynamics_forward(self, batch, stage, cfg):
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    loss_scope = cfg.wm.get('loss_scope', 'joint')
    if loss_scope not in ('low', 'high', 'joint'):
        raise ValueError(f'wm.loss_scope must be low|high|joint, got {loss_scope!r}')
    low_loss_enabled = loss_scope in ('low', 'joint')
    if 'emb' in batch:
        if low_loss_enabled or not hasattr(self, 'cached_embedding_dtype'):
            raise ValueError('Cached embeddings require verified frozen high-level training')
        output = {**batch, 'emb': batch['emb'].to(self.cached_embedding_dtype)}
    else:
        encode_batch = batch if low_loss_enabled else {key: value for (key, value) in batch.items() if key != 'action'}
        output = self.model.encode(encode_batch, return_visual_aux=False)
    emb = output['emb']
    raw_action = batch['action']
    k = cfg.wm.window_size
    alpha = cfg.wm.get('rollout_loss_alpha', 0.5)
    macro_action_alignment = cfg.wm.get('macro_action_alignment', 'endpoint_transition')
    lambda_H = float(cfg.loss.get('lambda_H', 1.0))
    lambda_sigreg = float(cfg.loss.get('weight', 0.3))
    lambda_sigreg_low = float(cfg.loss.get('low_sigreg_weight', lambda_sigreg))
    lambda_sigreg_high = float(cfg.loss.get('high_sigreg_weight', lambda_sigreg))
    lambda_sigreg_action = float(cfg.loss.get('macro_action_sigreg_weight', lambda_sigreg_high))
    if low_loss_enabled:
        act_emb = output['act_emb']
        (pred_list_L, tgt_list_L) = self.model.low_level_rollout_train(emb, act_emb, history_size=cfg.wm.history_size, num_rollout_steps=cfg.wm.num_rollout_steps)
        per_step_losses_L = [(p - t).pow(2).mean() for (p, t) in zip(pred_list_L, tgt_list_L)]
        output['L_loss'] = weighted_mean(per_step_losses_L, cfg.wm.rollout_loss_weighting, alpha)
        output['L_sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    high_level_enabled = cfg.wm.get('high_level_enabled', True)
    if loss_scope == 'high' and (not high_level_enabled):
        raise ValueError('loss_scope=high requires high_level_enabled=true')
    macro_action_regularization = emb.new_zeros(())
    if high_level_enabled:
        z_h_all = self.model.encode_high_level(emb, k)
        macro_encoder = self.model.macro_action_encoder
        macro_action_mode = getattr(macro_encoder, 'latent_mode', 'sigreg')
        if macro_action_mode == 'variational':
            (u_high, u_low, u_stats) = self.model.encode_macro_action(raw_action, k, return_u=True, return_stats=True, alignment=macro_action_alignment)
        else:
            (u_high, u_low) = self.model.encode_macro_action(raw_action, k, return_u=True, alignment=macro_action_alignment)
        M = cfg.wm.high_level_rollout_steps
        (pred_list_H, tgt_list_H) = self.model.high_level_rollout_train_from_latents(z_h_all, u_high, num_rollout_steps=M)
        per_step_losses_H = [(p - t).pow(2).mean() for (p, t) in zip(pred_list_H, tgt_list_H)]
        output['H_loss'] = weighted_mean(per_step_losses_H, cfg.wm.rollout_loss_weighting, alpha)
        output['H_sigreg_loss'] = self.sigreg(z_h_all.transpose(0, 1))
        valid_slice = slice(1, None) if macro_action_alignment == 'endpoint_transition' else slice(None)
        if macro_action_mode == 'sigreg':
            u_low_for_sigreg = u_low[:, valid_slice]
            output['A_sigreg_loss'] = self.sigreg(u_low_for_sigreg.transpose(0, 1))
            macro_action_regularization = lambda_sigreg_action * output['A_sigreg_loss']
        elif macro_action_mode == 'variational':
            kl_cfg = cfg.loss.get('macro_action_kl', {})
            free_bits = float(kl_cfg.get('free_bits', 0.0))
            (raw_kl, free_bits_kl) = diagonal_gaussian_kl(u_stats['mean'][:, valid_slice], u_stats['logvar'][:, valid_slice], free_bits=free_bits)
            beta = float(kl_cfg.get('beta', 0.001))
            warmup_epochs = int(kl_cfg.get('warmup_epochs', 0))
            if warmup_epochs > 0:
                beta *= min(1.0, (int(getattr(self, 'current_epoch', 0)) + 1) / warmup_epochs)
            output['A_kl_loss'] = raw_kl
            output['A_kl_free_bits_loss'] = free_bits_kl
            output['A_kl_beta'] = raw_kl.new_tensor(beta)
            macro_action_regularization = beta * free_bits_kl
        else:
            raise ValueError(f'Unknown macro-action latent mode {macro_action_mode!r}')
    low_objective = emb.new_zeros(())
    if low_loss_enabled:
        low_objective = output['L_loss'] + lambda_sigreg_low * output['L_sigreg_loss']
    if loss_scope == 'low':
        loss = low_objective
    elif loss_scope == 'high':
        loss = lambda_sigreg_high * output['H_sigreg_loss']
        loss += macro_action_regularization
        if 'H_loss' in output:
            loss += lambda_H * output['H_loss']
    else:
        loss = low_objective
        if 'H_loss' in output:
            loss += lambda_H * output['H_loss']
            loss += lambda_sigreg_high * output['H_sigreg_loss']
            loss += macro_action_regularization
    output['lambda_H'] = emb.new_tensor(lambda_H)
    output['loss'] = loss
    losses_dict = {f'{stage}/{k_}': v.detach() for (k_, v) in output.items() if 'loss' in k_ or k_ in ('A_kl_beta', 'lambda_H')}
    self.log_dict(losses_dict, on_step=True, on_epoch=True, sync_dist=stage != 'fit')
    return output

def load_init_weights(model, init_from, init_scope, cache_dir=None):
    if not init_from:
        return
    path = resolve_checkpoint_path(init_from, cache_dir)
    state = torch.load(path, map_location='cpu', weights_only=True)
    target = model.state_dict()
    if init_scope == 'low_only':
        state = {key: value for (key, value) in state.items() if key.startswith(LOW_LEVEL_MODULE_PREFIXES)}
        (state, _) = remap_checkpoint_encoder_keys(state, target)
        validate_low_only_checkpoint_keys(state, target)
        result = model.load_state_dict(state, strict=False)
        if any((key.startswith(LOW_LEVEL_MODULE_PREFIXES) for key in result.missing_keys)) or result.unexpected_keys:
            raise ValueError('Incomplete low-level checkpoint')
    elif init_scope == 'all':
        (state, _) = remap_checkpoint_encoder_keys(state, target)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError('init_from_scope must be low_only or all')

def build_optimizer_configs(cfg, total_steps):
    total_steps = max(2, int(total_steps))
    return {'main_opt': {'modules': 'model', 'optimizer': dict(cfg.optimizer), 'scheduler': {'type': 'LinearWarmupCosineAnnealingLR', 'warmup_steps': max(1, int(0.01 * total_steps)), 'max_steps': total_steps}, 'interval': 'step'}}

def training_step_budget(cfg, sample_count):
    devices = cfg.trainer.devices
    if isinstance(devices, (list, tuple)) or OmegaConf.is_list(devices):
        local_size = len(devices)
    elif str(devices) in ('auto', '-1'):
        local_size = 1 if cfg.trainer.accelerator == 'cpu' else max(1, torch.cuda.device_count())
    else:
        local_size = int(devices)
    world_size = int(os.environ.get('WORLD_SIZE', local_size * int(cfg.trainer.get('num_nodes', 1))))
    if world_size < 1:
        raise ValueError('Training requires a positive device count')
    per_rank = math.ceil(sample_count / world_size)
    batches = per_rank // cfg.loader.batch_size if cfg.loader.drop_last else math.ceil(per_rank / cfg.loader.batch_size)
    epochs = int(cfg.trainer.max_epochs)
    limit = int(cfg.trainer.get('max_steps', -1))
    total = epochs * batches if epochs >= 0 else limit
    if limit > 0:
        total = min(total, limit) if epochs >= 0 else limit
    if total < 1:
        raise ValueError('Training requires a finite positive optimizer-step budget and nonempty rank batches')
    return total

class DualWMTrainingModule(spt.Module):

    def __init__(self, gradient_clip_val=None, gradient_clip_algorithm='norm', **kwargs):
        super().__init__(**kwargs)
        self.manual_gradient_clip_val = gradient_clip_val
        self.manual_gradient_clip_algorithm = gradient_clip_algorithm

    def after_manual_backward(self):
        super().after_manual_backward()
        if self.manual_gradient_clip_val is None:
            return
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for optimizer in optimizers:
            self.clip_gradients(optimizer, gradient_clip_val=self.manual_gradient_clip_val, gradient_clip_algorithm=self.manual_gradient_clip_algorithm)

@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg):
    if cfg.stage not in ('low', 'high'):
        raise ValueError('Select stage=low or stage=high')
    if cfg.wm.rollout_loss_weighting != 'exponential' or not math.isfinite(cfg.wm.rollout_loss_alpha) or cfg.wm.rollout_loss_alpha <= 0:
        raise ValueError('Use exponential rollout weighting with a finite positive alpha')
    if cfg.wm.loss_scope not in ('low', 'high'):
        raise ValueError('Select a low_* or high_* two-stage training configuration')
    if cfg.wm.loss_scope == 'high' and (not cfg.init_from):
        raise ValueError('High-level training requires init_from=<low-level checkpoint>')
    (HS_L, N, k) = (cfg.wm.history_size, cfg.wm.num_rollout_steps, cfg.wm.window_size)
    (HS_H, M) = (cfg.wm.high_level_history_size, cfg.wm.high_level_rollout_steps)
    num_steps = HS_L + N if cfg.wm.loss_scope == 'low' else math.ceil(max(HS_L + N, k * (HS_H + M)) / k) * k
    OmegaConf.update(cfg, 'data.dataset.num_steps', num_steps, force_add=True)
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    for key in list(dataset_cfg):
        if key.endswith('_normalizer'):
            dataset_cfg.pop(key)
    if cfg.wm.loss_scope == 'high' and cfg.wm.macro_action_alignment == 'endpoint_transition':
        dataset_cfg['last_action_unused'] = True
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cfg.get('cache_dir'), **dataset_cfg)
    train_set, test_set = split_by_episode(dataset, cfg.train_split, cfg.seed)
    OmegaConf.update(cfg, 'data.split', {'unit': 'episode', 'seed': int(cfg.seed), 'train_episode_ids': list(train_set.episode_ids), 'test_episode_ids': list(test_set.episode_ids), 'train_windows': len(train_set), 'test_windows': len(test_set)}, force_add=True)
    transforms = [get_img_preprocessor('pixels', 'pixels', cfg.img_size)] if 'pixels' in cfg.data.dataset.keys_to_load else []
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if not col.startswith('pixels') and col != 'emb':
                transforms.append(get_column_normalizer(dataset, col, col, method=cfg.data.dataset.get(f'{col}_normalizer', 'zscore')))
        width = cfg.data.dataset.frameskip * dataset.get_dim('action')
        cfg.model.action_encoder.input_dim = width
        cfg.model.macro_action_encoder.input_dim = width
    dataset.transform = spt.data.transforms.Compose(*transforms)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader_cfg = dict(cfg.loader)
    val_workers = loader_cfg.pop('val_num_workers')
    if loader_cfg['num_workers'] == 0:
        loader_cfg.pop('prefetch_factor', None)
        loader_cfg.pop('multiprocessing_context', None)
        loader_cfg['persistent_workers'] = False
    train = torch.utils.data.DataLoader(train_set, **loader_cfg, generator=generator)
    val_cfg = {**loader_cfg, 'num_workers': val_workers, 'shuffle': False, 'drop_last': False}
    if val_workers == 0:
        val_cfg.pop('prefetch_factor', None)
        val_cfg.pop('multiprocessing_context', None)
        val_cfg['persistent_workers'] = False
    val = torch.utils.data.DataLoader(test_set, **val_cfg)
    pl.seed_everything(cfg.seed, workers=True)
    model = hydra.utils.instantiate(cfg.model)
    load_init_weights(model, cfg.init_from, cfg.init_from_scope, cfg.cache_dir)
    frozen = apply_train_control(model, cfg)
    cached_dtype = None
    if 'emb' in cfg.data.dataset.keys_to_load:
        import h5py
        from scripts.train.cache_encoder import encoder_fingerprint
        if cfg.wm.loss_scope != 'high' or not {'encoder', 'projector'}.issubset(frozen):
            raise ValueError('Cached embeddings require frozen encoder and projector')
        precision = str(cfg.trainer.precision)
        if precision not in ('bf16', 'bf16-mixed', '32', '32-true'):
            raise ValueError('Cached embeddings support bf16 mixed precision or float32 training')
        with h5py.File(dataset.h5_path, 'r') as cache:
            if not cache.attrs.get('complete') or cache.attrs.get('encoder_signature') != encoder_fingerprint(model) or int(cache.attrs.get('img_size', -1)) != cfg.img_size:
                raise ValueError('Frozen encoder cache does not match the initialized model and preprocessing')
            if int(cache.attrs.get('row_start', -1)) != 0 or int(cache.attrs.get('row_stop', -1)) != int(cache.attrs.get('source_rows', -2)):
                raise ValueError('Frozen encoder cache must cover the full source dataset')
            if cache.attrs.get('autocast_dtype') != ('bf16' if precision.startswith('bf16') else 'fp32'):
                raise ValueError('Frozen encoder cache precision differs from training precision')
            dtype_name = str(cache.attrs['embedding_dtype']).removeprefix('torch.')
            cached_dtype = getattr(torch, dtype_name)
    if cfg.wm.loss_scope == 'high':
        trainable = {name.split('.')[0] for (name, p) in model.named_parameters() if p.requires_grad}
        if trainable != {'temporal_encoder', 'macro_action_encoder', 'high_level_predictor'}:
            raise ValueError(f'Unexpected high-level optimizer scope: {trainable}')
    total_steps = training_step_budget(cfg, len(train_set))
    OmegaConf.update(cfg, 'optimizer_step_budget', total_steps, force_add=True)
    trainer_kwargs = dict(cfg.trainer)
    gradient_clip_val = trainer_kwargs.pop('gradient_clip_val', None)
    gradient_clip_algorithm = trainer_kwargs.pop('gradient_clip_algorithm', 'norm')
    module = DualWMTrainingModule(gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm, model=model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs), forward=partial(dual_wm_dynamics_forward, cfg=cfg), optim=build_optimizer_configs(cfg, total_steps))
    if cached_dtype is not None:
        module.cached_embedding_dtype = cached_dtype
    run_dir = swm.data.get_cache_dir(cfg.cache_dir, sub_folder='checkpoints') / cfg.output_model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'training.yaml', resolve=True)
    callbacks = [SaveCkptCallback(cfg.output_model_name, cfg.model, cache_dir=cfg.cache_dir)]
    if frozen:
        callbacks.append(KeepFrozenEvalCallback(frozen))
    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, num_sanity_val_steps=1, logger=False, enable_checkpointing=False, default_root_dir=str(run_dir))
    trainer.callbacks = [callback for callback in trainer.callbacks if not type(callback).__module__.startswith('stable_pretraining.')]
    trainer.fit(module, datamodule=spt.data.DataModule(train=train, val=val))
if __name__ == '__main__':
    run()
