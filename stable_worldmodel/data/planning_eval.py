from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler


TASK_ENVIRONMENTS = {
    'tworoom': 'swm/TwoRoom-v1',
    'reacher': 'swm/ReacherDMControl-v0',
    'pusht': 'swm/PushT-v1',
    'cube': 'swm/OGBCube-v0',
    'sokobanlong': 'swm/SokobanLong-v0',
}


@dataclass(frozen=True)
class PlanningEvaluationMetadata:
    task_name: str
    source_dataset_name: str
    task_seed: int
    sample_num_eval: int
    raw_goal_offset_steps: int
    evaluator_goal_offset_steps: int
    storage_mode: str
    episodes: dict[str, np.ndarray]
    tasks: dict[str, np.ndarray]
    normalizers: dict[str, dict]


def _attribute(group, name):
    if name not in group.attrs:
        raise ValueError(f'Dual-WM planning metadata is missing attribute {name!r}')
    value = group.attrs[name]
    return value.decode() if isinstance(value, bytes) else value


def _integer_attribute(group, name):
    value = _attribute(group, name)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'Dual-WM planning metadata attribute {name!r} must be an integer')
    return int(value)


def _integer_columns(group, names, size):
    result = {}
    for name in names:
        if name not in group or not isinstance(group[name], h5py.Dataset):
            raise ValueError(f'Dual-WM planning metadata is missing column {group.name}/{name}')
        values = group[name][()]
        if values.shape != (size,) or values.dtype.kind not in 'iu':
            raise ValueError(f'Dual-WM planning metadata column {name!r} must contain {size} integers')
        if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
            raise ValueError(f'Dual-WM planning metadata column {name!r} is out of bounds')
        result[name] = values.astype(np.int64, copy=False)
    return result


def _group(parent, name):
    if name not in parent or not isinstance(parent[name], h5py.Group):
        raise ValueError(f'Dual-WM planning metadata is missing group {parent.name}/{name}')
    return parent[name]


def _read_normalizers(group):
    result = {}
    for column, entry in group.items():
        if not isinstance(entry, h5py.Group):
            raise ValueError(f'Dual-WM planning normalizer {column!r} must be a group')
        dimension = _integer_attribute(entry, 'n_features_in')
        if dimension <= 0:
            raise ValueError(f'Dual-WM planning normalizer {column!r} has invalid feature count')
        with_mean = _attribute(entry, 'with_mean')
        with_std = _attribute(entry, 'with_std')
        if not isinstance(with_mean, (bool, np.bool_)) or not isinstance(with_std, (bool, np.bool_)) or not with_mean or not with_std:
            raise ValueError(f'Dual-WM planning normalizer {column!r} requires with_mean and with_std')
        values = {'n_features_in': dimension, 'with_mean': bool(with_mean), 'with_std': bool(with_std)}
        for name in ('mean', 'var', 'scale'):
            if name not in entry or not isinstance(entry[name], h5py.Dataset):
                raise ValueError(f'Dual-WM planning normalizer {column!r} is missing {name!r}')
            array = entry[name][()]
            if array.shape != (dimension,) or array.dtype.kind != 'f' or array.dtype.itemsize != 8 or not np.isfinite(array).all():
                raise ValueError(f'Dual-WM planning normalizer {column!r} has invalid {name!r}')
            if (name == 'var' and np.any(array < 0)) or (name == 'scale' and np.any(array <= 0)):
                raise ValueError(f'Dual-WM planning normalizer {column!r} has invalid {name!r}')
            values[name] = array
        if 'n_samples_seen' not in entry or not isinstance(entry['n_samples_seen'], h5py.Dataset):
            raise ValueError(f'Dual-WM planning normalizer {column!r} is missing n_samples_seen')
        count = entry['n_samples_seen'][()]
        if np.shape(count) not in ((), (dimension,)) or np.asarray(count).dtype.kind not in 'iuf' or not np.isfinite(count).all() or np.any(count <= 0):
            raise ValueError(f'Dual-WM planning normalizer {column!r} has invalid n_samples_seen')
        values['n_samples_seen'] = count
        values['source_dataset_name'] = str(_attribute(entry, 'source_dataset_name'))
        values['sklearn_version'] = str(_attribute(entry, 'sklearn_version'))
        result[column] = values
    return result


def read_planning_eval(dataset) -> PlanningEvaluationMetadata | None:
    path = dataset if isinstance(dataset, (str, Path)) else getattr(dataset, 'h5_path', None)
    if path is None:
        return None
    with h5py.File(path, 'r') as file:
        if '_planning_eval' not in file:
            return None
        group = _group(file, '_planning_eval')
        if _integer_attribute(group, 'schema_version') != 1:
            raise ValueError('Dual-WM planning metadata has an unsupported schema_version')
        task_seed = _integer_attribute(group, 'task_seed')
        size = _integer_attribute(group, 'sample_num_eval')
        raw_offset = _integer_attribute(group, 'raw_goal_offset_steps')
        evaluator_offset = _integer_attribute(group, 'evaluator_goal_offset_steps')
        storage_mode = str(_attribute(group, 'storage_mode'))
        if task_seed < 0 or size <= 0 or raw_offset <= 0 or evaluator_offset <= 0:
            raise ValueError('Dual-WM planning metadata has invalid task parameters')
        if storage_mode not in ('full_episodes', 'paired'):
            raise ValueError('Dual-WM planning metadata has an unsupported storage_mode')
        if (storage_mode == 'full_episodes' and raw_offset != evaluator_offset) or (storage_mode == 'paired' and evaluator_offset != 1):
            raise ValueError('Dual-WM planning metadata has inconsistent goal offsets')
        lengths = np.asarray(file['ep_len'][()])
        offsets = np.asarray(file['ep_offset'][()])
        if lengths.ndim != 1 or offsets.shape != lengths.shape or lengths.dtype.kind not in 'iu' or offsets.dtype.kind not in 'iu' or np.any(lengths <= 0):
            raise ValueError('Dual-WM planning dataset has invalid episode metadata')
        expected_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)))
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError('Dual-WM planning dataset episode offsets are inconsistent')
        episodes = _integer_columns(_group(group, 'episodes'), ('source_episode_idx', 'source_episode_offset', 'source_episode_length'), len(lengths))
        tasks = _integer_columns(_group(group, 'tasks'), ('source_row', 'source_episode_idx', 'source_start_step', 'local_row', 'local_episode_idx', 'local_start_step', 'environment_seed'), size)
        local_episodes = tasks['local_episode_idx']
        if np.any(local_episodes >= len(lengths)):
            raise ValueError('Dual-WM planning task episode is out of bounds')
        local_steps = tasks['local_start_step']
        if np.any(local_steps >= lengths[local_episodes] - evaluator_offset):
            raise ValueError('Dual-WM planning task start/goal window is out of bounds')
        if not np.array_equal(tasks['local_row'], offsets[local_episodes] + local_steps):
            raise ValueError('Dual-WM planning task row does not match its local episode/start')
        if not np.array_equal(tasks['source_episode_idx'], episodes['source_episode_idx'][local_episodes]):
            raise ValueError('Dual-WM planning task source episode mapping is inconsistent')
        source_steps = tasks['source_start_step']
        if np.any(source_steps >= episodes['source_episode_length'][local_episodes]):
            raise ValueError('Dual-WM planning task source start is out of bounds')
        if not np.array_equal(tasks['source_row'], episodes['source_episode_offset'][local_episodes] + source_steps):
            raise ValueError('Dual-WM planning task source row mapping is inconsistent')
        if storage_mode == 'full_episodes' and (not np.array_equal(lengths, episodes['source_episode_length']) or not np.array_equal(local_steps, source_steps)):
            raise ValueError('Dual-WM planning full episodes must preserve lengths and start steps')
        episode_column = 'episode_idx' if 'episode_idx' in file else 'ep_idx'
        if episode_column not in file or 'step_idx' not in file:
            raise ValueError('Dual-WM planning dataset is missing episode/step columns')
        total_rows = int(np.sum(lengths, dtype=np.int64))
        for column in (episode_column, 'step_idx'):
            if not isinstance(file[column], h5py.Dataset) or file[column].shape != (total_rows,):
                raise ValueError(f'Dual-WM planning dataset has invalid {column!r}')
        row_episodes = np.asarray([file[episode_column][int(row)] for row in tasks['local_row']])
        row_steps = np.asarray([file['step_idx'][int(row)] for row in tasks['local_row']])
        if not np.array_equal(row_episodes, local_episodes) or not np.array_equal(row_steps, local_steps):
            raise ValueError('Dual-WM planning task row disagrees with episode/step columns')
        if 'seed' in file:
            row_seeds = np.asarray([file['seed'][int(row)] for row in tasks['local_row']])
            if not np.array_equal(row_seeds, tasks['environment_seed']):
                raise ValueError('Dual-WM planning task seed disagrees with the dataset seed')
        return PlanningEvaluationMetadata(str(_attribute(group, 'task_name')), str(_attribute(group, 'source_dataset_name')), task_seed, size, raw_offset, evaluator_offset, storage_mode, episodes, tasks, _read_normalizers(_group(group, 'normalizers')))


def select_planning_tasks(metadata, *, seed, sample_num_eval, task_offset, num_eval, goal_offset_steps):
    if int(seed) != metadata.task_seed:
        raise ValueError('Dual-WM planning task seed does not match the fixed task pool')
    if int(sample_num_eval) != metadata.sample_num_eval:
        raise ValueError('Dual-WM planning sample_num_eval does not match the fixed task pool')
    if int(goal_offset_steps) != metadata.evaluator_goal_offset_steps:
        raise ValueError('Dual-WM planning evaluator goal offset does not match the fixed tasks')
    offset, count = int(task_offset), int(num_eval)
    if offset < 0 or count <= 0 or offset + count > metadata.sample_num_eval:
        raise ValueError('Invalid Dual-WM planning task slice')
    selection = slice(offset, offset + count)
    return tuple(metadata.tasks[column][selection].copy() for column in ('local_row', 'local_episode_idx', 'local_start_step'))


def restore_planning_scalers(metadata, columns):
    process = {}
    for column in columns:
        if column == 'pixels':
            continue
        if column not in metadata.normalizers:
            raise ValueError(f'Dual-WM planning metadata is missing normalizer {column!r}')
        values = metadata.normalizers[column]
        scaler = StandardScaler(with_mean=values['with_mean'], with_std=values['with_std'])
        scaler.mean_ = values['mean'].copy()
        scaler.var_ = values['var'].copy()
        scaler.scale_ = values['scale'].copy()
        scaler.n_samples_seen_ = values['n_samples_seen'].copy()
        scaler.n_features_in_ = values['n_features_in']
        process[column] = scaler
        if column != 'action':
            process[f'goal_{column}'] = scaler
    return process


def validate_planning_environment_seeds(expected, actual):
    expected_values = np.asarray(expected, dtype=np.int64)
    actual_values = np.asarray(actual)
    if actual_values.shape != expected_values.shape or actual_values.dtype.kind not in 'iu' or not np.array_equal(actual_values, expected_values):
        raise ValueError('Dual-WM planning environment seeds do not match the fixed tasks')


def validate_planning_task_environment(metadata, environment_name):
    expected = TASK_ENVIRONMENTS.get(metadata.task_name)
    if expected is None or str(environment_name) != expected:
        raise ValueError(f'Dual-WM planning dataset task {metadata.task_name!r} does not match environment {environment_name!r}')
