from __future__ import annotations

import argparse
import ctypes
import errno
import json
import math
import os
import re
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np
import sklearn
import yaml
from sklearn.preprocessing import StandardScaler


TASK_NAMES = ('tworoom', 'reacher', 'pusht', 'cube', 'sokobanlong')
METADATA_GROUP = '_planning_eval'
COPY_BYTES = 32 * 1024 * 1024
CHUNK_BYTES = 2 * 1024 * 1024


def report(event: str, **values) -> None:
    print(json.dumps({'event': event, **values}, sort_keys=True), flush=True)


def integer_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a one-dimensional integer array')
    if array.dtype.kind == 'u' and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f'{name} contains an unsupported integer')
    return array.astype(np.int64, copy=False)


def safe_attribute(name: str, value) -> bool:
    if re.search(r'path|directory|checkpoint|result|outcome|success|reward|score|accuracy|machine|hostname|username', name, re.IGNORECASE):
        return False
    array = np.asarray(value)
    if array.dtype.kind in 'biufc':
        return True
    if array.dtype.kind not in 'OSU':
        return False
    for item in array.reshape(-1):
        if isinstance(item, bytes):
            item = item.decode('utf-8', errors='replace')
        if not isinstance(item, str):
            return False
        if re.search(r'(^|\s)(/|~[/\\]|[A-Za-z]:[/\\])|file://|/home/|/data\d*/|\\Users\\', item):
            return False
        if re.search(r'checkpoint|outcome|success.rate|reward|accuracy|hostname|username', item, re.IGNORECASE):
            return False
    return True


def copy_attributes(source, target) -> None:
    for name, value in source.attrs.items():
        if safe_attribute(name, value):
            target.attrs[name] = value


def source_layout(source: h5py.File) -> tuple[np.ndarray, np.ndarray, list[str]]:
    for name in ('ep_len', 'ep_offset', 'step_idx'):
        if name not in source or not isinstance(source.get(name, getlink=True), h5py.HardLink):
            raise ValueError(f'Source requires an ordinary {name} dataset')
    lengths = integer_vector(source['ep_len'][:], 'ep_len')
    offsets = integer_vector(source['ep_offset'][:], 'ep_offset')
    if not len(lengths) or len(lengths) != len(offsets) or np.any(lengths <= 0):
        raise ValueError('Source episode lengths and offsets are invalid')
    expected = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths[:-1], dtype=np.int64)))
    if not np.array_equal(offsets, expected):
        raise ValueError('Source episodes must cover consecutive rows from zero')
    row_count = int(offsets[-1] + lengths[-1])
    columns = []
    for name in source:
        if name in ('ep_len', 'ep_offset'):
            continue
        if not isinstance(source.get(name, getlink=True), h5py.HardLink):
            raise ValueError(f'Source links are unsupported: {name}')
        field = source[name]
        if not isinstance(field, h5py.Dataset) or field.ndim < 1 or field.shape[0] != row_count:
            raise ValueError(f'Source field must contain one value per row: {name}')
        if h5py.check_dtype(ref=field.dtype) is not None:
            raise ValueError(f'Source object references are unsupported: {name}')
        columns.append(name)
    if not {'ep_idx', 'episode_idx'}.intersection(columns):
        raise ValueError('Source requires ep_idx or episode_idx')
    integer_vector(source['step_idx'][:], 'step_idx')
    return lengths, offsets, columns


def benchmark_tasks(benchmark: Path, task: str, source: h5py.File, lengths: np.ndarray, offsets: np.ndarray, seed: int, evaluator_offset: int) -> dict[str, np.ndarray]:
    specification = json.loads(benchmark.read_text())
    if specification.get('task') != task:
        raise ValueError('Benchmark task does not match --task')
    if specification.get('num_tasks') != 200 or specification.get('task_seed') != seed:
        raise ValueError('Benchmark must contain 200 tasks with the requested task seed')
    if specification.get('offset_env_steps') != 100:
        raise ValueError('Benchmark must use a raw goal offset of 100 steps')
    if specification.get('evaluator_goal_offset_steps', evaluator_offset) != evaluator_offset:
        raise ValueError('Benchmark evaluator goal offset does not match the storage mode')
    rows = integer_vector(specification.get('dataset_rows'), 'dataset_rows')
    seeds = integer_vector(specification.get('environment_seeds'), 'environment_seeds')
    if len(rows) != 200 or len(seeds) != 200 or len(np.unique(rows)) != 200:
        raise ValueError('Benchmark requires 200 unique dataset rows and 200 environment seeds')
    if np.any(seeds < 0):
        raise ValueError('Benchmark environment seeds must be nonnegative')
    if np.any(rows < 0) or np.any(rows >= offsets[-1] + lengths[-1]):
        raise ValueError('Benchmark row lies outside the source dataset')
    episodes = np.searchsorted(offsets, rows, side='right') - 1
    steps = rows - offsets[episodes]
    if np.any(steps + evaluator_offset >= lengths[episodes]):
        raise ValueError('Benchmark start/goal pair crosses an episode boundary')
    order = np.argsort(rows)
    recorded_steps = integer_vector(source['step_idx'][rows[order]], 'benchmark step_idx')
    if not np.array_equal(recorded_steps, steps[order]):
        raise ValueError('Source step_idx differs from the benchmark episode-relative start')
    for name in ('ep_idx', 'episode_idx'):
        if name in source:
            recorded_episodes = integer_vector(source[name][rows[order]], name)
            if not np.array_equal(recorded_episodes, episodes[order]):
                raise ValueError(f'Source {name} differs from ep_len/ep_offset episode indices')
    if 'seed' in source:
        recorded_seeds = integer_vector(source['seed'][rows[order]], 'seed')
        if not np.array_equal(recorded_seeds, seeds[order]):
            raise ValueError('Source seed differs from the benchmark environment seed')
    return {'source_row': rows, 'source_episode_idx': episodes, 'source_start_step': steps, 'environment_seed': seeds}


def select_episodes(lengths: np.ndarray, required: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('--episode-fraction must be greater than zero and at most one')
    count = math.ceil(len(lengths) * fraction)
    required = np.unique(required)
    if len(required) > count:
        raise ValueError('Requested episode fraction cannot contain every benchmark task')
    remaining = np.setdiff1d(np.arange(len(lengths), dtype=np.int64), required, assume_unique=True)
    extra = np.random.default_rng(seed).choice(remaining, size=count - len(required), replace=False)
    return np.sort(np.concatenate((required, extra)))


def fit_normalizers(stats_source: Path, columns: list[str]) -> dict[str, StandardScaler]:
    normalizers = {}
    with h5py.File(stats_source, 'r') as source:
        for name in columns:
            if name == 'pixels':
                continue
            if name not in source or not isinstance(source[name], h5py.Dataset):
                raise ValueError(f'Statistics source does not contain {name}')
            values = source[name][:]
            if values.ndim != 2 or values.dtype.kind not in 'iuf':
                raise ValueError(f'Statistics column must be a numeric matrix: {name}')
            values = values[~np.isnan(values).any(axis=1)]
            if not len(values):
                raise ValueError(f'Statistics column has no valid rows: {name}')
            normalizers[name] = StandardScaler().fit(values)
            report('normalizer_fitted', column=name, samples=int(normalizers[name].n_samples_seen_), features=int(normalizers[name].n_features_in_))
    return normalizers


def write_metadata(output: h5py.File, task: str, source: Path, stats_source: Path, seed: int, selected: np.ndarray, lengths: np.ndarray, offsets: np.ndarray, local_offsets: np.ndarray, tasks: dict[str, np.ndarray], normalizers: dict[str, StandardScaler], mode: str, evaluator_offset: int) -> None:
    metadata = output.create_group(METADATA_GROUP)
    metadata.attrs.update(schema_version=1, task_name=task, source_dataset_name=source.name, task_seed=seed, sample_num_eval=200, raw_goal_offset_steps=100, evaluator_goal_offset_steps=evaluator_offset, storage_mode=mode)
    episodes = metadata.create_group('episodes')
    for name, values in (('source_episode_idx', selected), ('source_episode_offset', offsets[selected]), ('source_episode_length', lengths[selected])):
        episodes.create_dataset(name, data=values, dtype=np.int64)
    tasks = dict(tasks)
    tasks['local_episode_idx'] = np.searchsorted(selected, tasks['source_episode_idx'])
    tasks['local_start_step'] = tasks['source_start_step'].copy()
    tasks['local_row'] = local_offsets[tasks['local_episode_idx']] + tasks['local_start_step']
    task_group = metadata.create_group('tasks')
    for name, values in tasks.items():
        task_group.create_dataset(name, data=values, dtype=np.int64)
    normalizer_group = metadata.create_group('normalizers')
    for name, scaler in normalizers.items():
        group = normalizer_group.create_group(name)
        group.attrs.update(n_features_in=int(scaler.n_features_in_), with_mean=bool(scaler.with_mean), with_std=bool(scaler.with_std), source_dataset_name=stats_source.name, sklearn_version=sklearn.__version__)
        for stored, attribute in (('mean', 'mean_'), ('var', 'var_'), ('scale', 'scale_')):
            group.create_dataset(stored, data=np.asarray(getattr(scaler, attribute), dtype=np.float64))
        group.create_dataset('n_samples_seen', data=np.asarray(scaler.n_samples_seen_))


def create_row_dataset(output: h5py.File, name: str, source: h5py.Dataset, row_count: int, compression: str) -> h5py.Dataset:
    row_bytes = max(1, int(np.prod(source.shape[1:], dtype=np.int64)) * source.dtype.itemsize)
    chunk_rows = min(row_count, max(1, CHUNK_BYTES // row_bytes), 8 if source.ndim >= 3 else 8192)
    options = {'compression': 'lzf'} if compression == 'lzf' else dict(hdf5plugin.Blosc(cname='lz4', clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
    if source.dtype.kind == 'O':
        options = {'compression': 'lzf'}
    target = output.create_dataset(name, shape=(row_count, *source.shape[1:]), dtype=source.dtype, chunks=(chunk_rows, *source.shape[1:]), **options)
    copy_attributes(source, target)
    return target


def copy_rows(source: h5py.File, output: h5py.File, columns: list[str], selected: np.ndarray, lengths: np.ndarray, offsets: np.ndarray, local_offsets: np.ndarray, compression: str) -> None:
    row_count = int(lengths[selected].sum())
    runs = np.split(selected, np.flatnonzero(np.diff(selected) != 1) + 1)
    for name in columns:
        field = source[name]
        target = create_row_dataset(output, name, field, row_count, compression)
        if name in ('ep_idx', 'episode_idx'):
            for local_episode, source_episode in enumerate(selected):
                start = int(local_offsets[local_episode])
                target[start:start + int(lengths[source_episode])] = local_episode
            report('column_copied', column=name, rows=row_count)
            continue
        row_bytes = max(1, int(np.prod(field.shape[1:], dtype=np.int64)) * field.dtype.itemsize)
        block_rows = max(1, COPY_BYTES // row_bytes)
        copied = 0
        next_progress = 0.1
        for run in runs:
            source_start = int(offsets[run[0]])
            source_end = int(offsets[run[-1]] + lengths[run[-1]])
            local_episode = int(np.searchsorted(selected, run[0]))
            local_start = int(local_offsets[local_episode])
            for begin in range(source_start, source_end, block_rows):
                end = min(begin + block_rows, source_end)
                destination = local_start + begin - source_start
                values = field[begin:end]
                if name == 'step_idx':
                    episode_indices = np.searchsorted(offsets, np.arange(begin, end), side='right') - 1
                    expected = np.arange(begin, end) - offsets[episode_indices]
                    if not np.array_equal(values, expected):
                        raise ValueError('Selected source step_idx is not episode-relative')
                target[destination:destination + end - begin] = values
                copied += end - begin
                if name == 'pixels' and copied / row_count >= next_progress:
                    report('copy_progress', column=name, rows=copied, total_rows=row_count)
                    next_progress += 0.1
        report('column_copied', column=name, rows=copied)


def publish(partial: Path, output: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is not None:
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(partial), -100, os.fsencode(output), 1) == 0:
            return
        error = ctypes.get_errno()
        if error not in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
            raise OSError(error, os.strerror(error), str(output))
    os.link(partial, output)
    partial.unlink()


def build_eval_data(*, task: str, source: Path, benchmark: Path, output: Path, stats_source: Path | None = None, episode_fraction: float = 0.1, seed: int = 42, mode: str = 'full_episodes', compression: str = 'blosc', normalizer_columns: list[str] | None = None) -> dict:
    if task not in TASK_NAMES or mode not in ('full_episodes', 'paired') or compression not in ('blosc', 'lzf'):
        raise ValueError('Unsupported Dual-WM task, storage mode, or compression')
    if mode == 'paired' and (task != 'sokobanlong' or episode_fraction != 1.0):
        raise ValueError('Paired Sokoban storage requires --task sokobanlong and explicit --episode-fraction 1.0')
    if task == 'sokobanlong' and mode != 'paired':
        raise ValueError('Sokoban full episodes require canonical zero-based task conversion; currently supported storage is paired')
    if task == 'sokobanlong' and stats_source is None:
        raise ValueError('Sokoban requires an explicit --stats-source for its original full statistics corpus')
    source, benchmark, output = (Path(path).expanduser() for path in (source, benchmark, output))
    stats_source = Path(stats_source).expanduser() if stats_source is not None else source
    if task == 'sokobanlong' and stats_source.resolve() == source.resolve():
        raise ValueError('Sokoban --stats-source must be separate from its evaluation source')
    partial = output.with_name(output.name + '.partial')
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(f'Refusing to overwrite an existing output or partial file: {output}')
    if normalizer_columns is None:
        configuration = yaml.safe_load((Path(__file__).parent / 'config' / f'{task}.yaml').read_text())
        normalizer_columns = list(configuration['dataset']['keys_to_cache'])
    with h5py.File(source, 'r', rdcc_nbytes=64 * 1024 * 1024) as original:
        lengths, offsets, columns = source_layout(original)
        evaluator_offset = 1 if mode == 'paired' else 100
        if mode == 'paired' and (len(lengths) != 200 or np.any(lengths != 2)):
            raise ValueError('Paired Sokoban source must contain exactly 200 two-row episodes')
        tasks = benchmark_tasks(benchmark, task, original, lengths, offsets, seed, evaluator_offset)
        selected = select_episodes(lengths, tasks['source_episode_idx'], episode_fraction, seed)
        selected_lengths = lengths[selected]
        local_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(selected_lengths[:-1], dtype=np.int64)))
        row_count = int(selected_lengths.sum())
        report('build_started', task=task, source_dataset_name=source.name, selected_episodes=len(selected), required_episodes=len(np.unique(tasks['source_episode_idx'])), total_source_episodes=len(lengths), selected_rows=row_count, episode_fraction=episode_fraction, storage_mode=mode)
        normalizers = fit_normalizers(stats_source, normalizer_columns)
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(partial, 'x', libver=('earliest', 'v110')) as target:
            copy_attributes(original, target)
            target.create_dataset('ep_len', data=selected_lengths, dtype=original['ep_len'].dtype)
            target.create_dataset('ep_offset', data=local_offsets, dtype=original['ep_offset'].dtype)
            copy_attributes(original['ep_len'], target['ep_len'])
            copy_attributes(original['ep_offset'], target['ep_offset'])
            write_metadata(target, task, source, stats_source, seed, selected, lengths, offsets, local_offsets, tasks, normalizers, mode, evaluator_offset)
            copy_rows(original, target, columns, selected, lengths, offsets, local_offsets, compression)
            target.flush()
    with partial.open('rb') as stream:
        os.fsync(stream.fileno())
    publish(partial, output)
    receipt = {'task': task, 'output_dataset_name': output.name, 'selected_episodes': len(selected), 'selected_rows': row_count, 'sample_num_eval': len(tasks['source_row']), 'size_bytes': output.stat().st_size, 'sklearn_version': sklearn.__version__, 'storage_mode': mode, 'episode_fraction': episode_fraction}
    report('build_completed', **receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description='Build independent Dual-WM planning datasets while preserving canonical tasks and full-corpus normalizers.')
    parser.add_argument('--task', choices=TASK_NAMES, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stats-source', type=Path)
    parser.add_argument('--benchmark', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', choices=('full_episodes', 'paired'), default='full_episodes')
    parser.add_argument('--compression', choices=('blosc', 'lzf'), default='blosc')
    arguments = parser.parse_args()
    benchmark = arguments.benchmark or Path(__file__).parent / 'task_definitions' / f'{arguments.task}_o100.json'
    build_eval_data(task=arguments.task, source=arguments.source, stats_source=arguments.stats_source, benchmark=benchmark, output=arguments.output, episode_fraction=arguments.episode_fraction, seed=arguments.seed, mode=arguments.mode, compression=arguments.compression)


if __name__ == '__main__':
    main()
