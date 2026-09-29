import json
import os
import time
from copy import deepcopy
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.data.utils import load_dataset
from stable_worldmodel.data.planning_eval import read_planning_eval, restore_planning_scalers, select_planning_tasks, validate_planning_environment_seeds, validate_planning_task_environment

def img_transform(cfg, dtype=torch.float32):
    normalization = cfg.eval.get('image_normalization', 'imagenet')
    if normalization == 'imagenet':
        stats = spt.data.dataset_stats.ImageNet
    elif normalization == 'half':
        stats = {'mean': [0.5] * 3, 'std': [0.5] * 3}
    elif normalization in (None, 'none'):
        stats = None
    else:
        raise ValueError(f'Unknown eval.image_normalization={normalization!r}')
    operations = [transforms.ToImage(), transforms.ToDtype(dtype, scale=True)]
    if stats is not None:
        operations.append(transforms.Normalize(**stats))
    operations.append(transforms.Resize(size=cfg.eval.img_size))
    transform = transforms.Compose(operations)
    return transform

def get_episodes_length(dataset, episodes):
    col_name = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    episode_idx = np.asarray(dataset.get_col_data(col_name))
    step_idx = np.asarray(dataset.get_col_data('step_idx'))
    ids, inverse = np.unique(episode_idx, return_inverse=True)
    lengths = np.zeros(len(ids), dtype=np.int64)
    np.maximum.at(lengths, inverse, step_idx + 1)
    positions = np.searchsorted(ids, episodes)
    if np.any(positions >= len(ids)) or not np.array_equal(ids[positions], episodes):
        raise ValueError('Requested episode has no dataset rows')
    return lengths[positions]

def apply_subgoal_cost_overrides(model, cfg):
    subgoal_cost = cfg.get('subgoal_cost')
    if subgoal_cost is None:
        return
    mode = subgoal_cost.get('mode', 'dynamic_window_aligned')
    model.subgoal_cost_mode = mode

def apply_high_level_goal_cost_overrides(model, cfg):
    high_level_goal_cost = cfg.get('high_level_goal_cost')
    if high_level_goal_cost is None:
        return
    mode = high_level_goal_cost.get('mode', 'compatibility')
    model.high_level_goal_cost_mode = mode

def select_tasks(dataset, cfg, planning_eval=None):
    planning_eval = planning_eval if planning_eval is not None else read_planning_eval(dataset)
    if planning_eval is not None:
        return select_planning_tasks(planning_eval, seed=cfg.seed, sample_num_eval=cfg.eval.sample_num_eval, task_offset=cfg.eval.task_offset, num_eval=cfg.eval.num_eval, goal_offset_steps=cfg.eval.goal_offset_steps)
    col = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    episode_idx = np.asarray(dataset.get_col_data(col))
    episode_ids = np.unique(episode_idx)
    lengths = get_episodes_length(dataset, episode_ids)
    max_start = lengths - cfg.eval.goal_offset_steps - 1
    valid = np.flatnonzero(dataset.get_col_data('step_idx') <= max_start[np.searchsorted(episode_ids, episode_idx)])
    sample_size = int(cfg.eval.sample_num_eval)
    offset = int(cfg.eval.task_offset)
    count = int(cfg.eval.num_eval)
    if count <= 0 or sample_size <= 0 or offset < 0 or (offset + count > sample_size):
        raise ValueError('Invalid evaluation task slice')
    if sample_size > len(valid):
        raise ValueError('Dataset has fewer valid start/goal pairs than sample_num_eval')
    rng = np.random.default_rng(cfg.seed)
    rows = np.sort(valid[rng.choice(len(valid), size=sample_size, replace=False)])[offset:offset + count]
    data = dataset.get_row_data(rows)
    return (rows, data[col], data['step_idx'])

def prepare_scalers(dataset, stats_dataset, cfg, planning_eval=None):
    planning_eval = planning_eval if planning_eval is not None else read_planning_eval(dataset)
    if planning_eval is not None:
        return restore_planning_scalers(planning_eval, cfg.dataset.keys_to_cache)
    if stats_dataset is None:
        stats_dataset = dataset if cfg.dataset.stats == cfg.eval.dataset_name else load_dataset(cfg.dataset.stats, cache_dir=cfg.cache_dir, keys_to_cache=list(cfg.dataset.keys_to_cache))
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == 'pixels':
            continue
        values = stats_dataset.get_col_data(col)
        values = values[~np.isnan(values).any(axis=1)]
        process[col] = preprocessing.StandardScaler().fit(values)
        if col != 'action':
            process[f'goal_{col}'] = process[col]
    return process

@hydra.main(version_base=None, config_path='./config', config_name='tworoom')
def run(cfg):
    target = Path(cfg.output.filename).expanduser()
    if target.exists():
        raise FileExistsError(f'Refusing to replace evaluation results: {target}')
    if cfg.plan_config.horizon * cfg.plan_config.action_block > cfg.eval.eval_budget:
        raise ValueError('Planning horizon exceeds evaluation budget')
    device = str(cfg.solver.device)
    if device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA is required by the selected evaluation configuration')
    dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {'pixels': img_transform(cfg, dtype), 'goal': img_transform(cfg, dtype)}
    dataset = load_dataset(cfg.eval.dataset_name, cache_dir=cfg.cache_dir, keys_to_cache=list(cfg.dataset.keys_to_cache))
    planning_eval = read_planning_eval(dataset)
    if planning_eval is not None:
        validate_planning_task_environment(planning_eval, cfg.world.env_name)
    process = prepare_scalers(dataset, None, cfg, planning_eval)
    (rows, episodes, start_steps) = select_tasks(dataset, cfg, planning_eval)
    task_slice = slice(int(cfg.eval.task_offset), int(cfg.eval.task_offset) + int(cfg.eval.num_eval))
    fixed_environment_seeds = planning_eval.tasks['environment_seed'][task_slice] if planning_eval is not None else None
    checkpoint = Path(cfg.policy).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parents[2] / checkpoint
    model = swm.wm.utils.load_pretrained(checkpoint).to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    apply_subgoal_cost_overrides(model, cfg)
    apply_high_level_goal_cost_overrides(model, cfg)
    if cfg.get('compile', False):
        model.encoder = torch.compile(model.encoder)
        model.predictor = torch.compile(model.predictor)
    world_kwargs = OmegaConf.to_container(cfg.world, resolve=True)
    world_kwargs['max_episode_steps'] = 2 * cfg.eval.eval_budget
    batch_size = min(int(cfg.eval.env_batch_size), int(cfg.eval.num_eval))
    if batch_size <= 0:
        raise ValueError('env_batch_size must be positive')
    successes = []
    initial_successes = []
    environment_seeds = []
    world = None
    policy = None
    start = time.perf_counter()
    try:
        for begin in range(0, cfg.eval.num_eval, batch_size):
            end = min(begin + batch_size, cfg.eval.num_eval)
            size = end - begin
            created_world = world is None or world.num_envs != size or (not cfg.eval.reuse_env_batches)
            if created_world:
                if world is not None:
                    world.close()
                kwargs = {**world_kwargs, 'num_envs': size}
                world = swm.World(**kwargs, image_shape=tuple(cfg.eval.get('render_image_shape', [224, 224])))
            if created_world or (begin > 0 and cfg.eval.get('reset_solver_rng_per_batch', False)):
                solver = hydra.utils.instantiate(cfg.solver, model=model)
                policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(**cfg.plan_config), process=process, transform=transform)
                world.set_policy(policy)
            else:
                policy.set_env(world.envs)
            policy._next_init = None
            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', dtype=torch.bfloat16, enabled=cfg.get('bf16', False)):
                reset_seed = fixed_environment_seeds[begin:end].tolist() if fixed_environment_seeds is not None else int(cfg.seed) + int(cfg.eval.task_offset) + begin
                metrics = world.evaluate(dataset=dataset, seed=reset_seed, start_steps=start_steps[begin:end].tolist(), goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget, episodes_idx=episodes[begin:end].tolist(), callables=OmegaConf.to_container(cfg.eval.callables, resolve=True))
            if fixed_environment_seeds is not None:
                validate_planning_environment_seeds(fixed_environment_seeds[begin:end], metrics['seeds'])
            successes.extend(np.asarray(metrics['episode_successes'], dtype=bool).tolist())
            initial_successes.extend(metrics.get('initial_successes', []))
            environment_seeds.extend(np.asarray(metrics['seeds'], dtype=np.int64).tolist())
            print(json.dumps({'tasks_completed': len(successes), 'tasks_total': int(cfg.eval.num_eval), 'success_count': int(sum(successes))}), flush=True)
    finally:
        if world is not None:
            world.close()
    result = {'success_rate': float(np.mean(successes) * 100), 'success_count': int(sum(successes)), 'num_tasks': len(successes), 'episode_successes': successes, 'dataset_rows': rows.tolist(), 'evaluation_time_s': time.perf_counter() - start, 'config': OmegaConf.to_container(cfg, resolve=True)}
    result['environment_seeds'] = environment_seeds
    if planning_eval is not None:
        result['source_dataset_rows'] = planning_eval.tasks['source_row'][task_slice].tolist()
        result['source_episode_indices'] = planning_eval.tasks['source_episode_idx'][task_slice].tolist()
        result['source_start_steps'] = planning_eval.tasks['source_start_step'][task_slice].tolist()
        result['planning_episode_indices'] = episodes.tolist()
        result['planning_start_steps'] = start_steps.tolist()
        result['source_dataset_name'] = planning_eval.source_dataset_name
        result['planning_dataset_storage_mode'] = planning_eval.storage_mode
        result['raw_goal_offset_steps'] = planning_eval.raw_goal_offset_steps
        result['planning_eval_schema_version'] = 1
    result['reset_solver_rng_per_batch'] = bool(cfg.eval.get('reset_solver_rng_per_batch', False))
    if initial_successes:
        result['initial_success_count'] = int(sum(initial_successes))
        result['added_solve_count'] = int(sum((s and (not initial) for (s, initial) in zip(successes, initial_successes))))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print({k: result[k] for k in ('success_rate', 'success_count', 'num_tasks')})
if __name__ == '__main__':
    run()
