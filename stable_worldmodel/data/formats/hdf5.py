from __future__ import annotations
import ctypes
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
import h5py
import hdf5plugin
import numpy as np
import torch
from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import Format, register_format, validate_write_mode
from stable_worldmodel.data.utils import get_cache_dir

class HDF5Dataset(Dataset):

    def __init__(self, name: str | None=None, frameskip: int=1, num_steps: int=1, transform: Callable[[dict], dict] | None=None, keys_to_load: list[str] | None=None, keys_to_cache: list[str] | None=None, keys_to_merge: dict[str, list[str] | str] | None=None, cache_dir: str | Path | None=None, path: str | Path | None=None, last_action_unused: bool=False) -> None:
        if path is not None:
            self.h5_path = Path(path)
        else:
            if name is None:
                raise TypeError('HDF5Dataset requires either `name` or `path`')
            datasets_dir = get_cache_dir(cache_dir, sub_folder='datasets')
            self.h5_path = Path(datasets_dir, f'{name}.h5')
        self.h5_file: h5py.File | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._blosc_lib = None
        self.last_action_unused = bool(last_action_unused)
        with self._open_h5() as f:
            lengths, offsets = (f['ep_len'][:], f['ep_offset'][:])
            self._keys = keys_to_load or [k for k, value in f.items() if isinstance(value, h5py.Dataset) and k not in ('ep_len', 'ep_offset')]
            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")
        super().__init__(lengths, offsets, frameskip, num_steps, transform)
        if self.last_action_unused:
            required = (num_steps - 1) * frameskip + 1
            self.clip_indices = [(ep, start) for ep, length in enumerate(lengths) if length >= required for start in range(length - required + 1)]
        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open_h5(self) -> h5py.File:
        return h5py.File(self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024)

    def _open(self) -> None:
        if self.h5_file is None:
            self.h5_file = self._open_h5()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['h5_file'] = None
        state['_blosc_lib'] = None
        return state

    def _get_blosc_lib(self):
        if self._blosc_lib is False:
            return None
        if self._blosc_lib is None:
            try:
                path = os.path.join(hdf5plugin.PLUGINS_PATH, 'libh5blosc.so')
                lib = ctypes.CDLL(path)
                lib.blosc_decompress.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
                lib.blosc_decompress.restype = ctypes.c_int
                lib.blosc_set_nthreads(1)
                self._blosc_lib = lib
            except (AttributeError, OSError):
                self._blosc_lib = False
        return self._blosc_lib or None

    def _can_direct_read_blosc_pixels(self) -> bool:
        if 'pixels' not in self._keys:
            return False
        self._open()
        pixels = self.h5_file['pixels']
        if pixels.ndim != 4 or pixels.chunks is None:
            return False
        if tuple(pixels.chunks[1:]) != tuple(pixels.shape[1:]):
            return False
        plist = pixels.id.get_create_plist()
        if plist.get_nfilters() != 1:
            return False
        filter_id, *_ = plist.get_filter(0)
        return filter_id == hdf5plugin.Blosc.filter_id and self._get_blosc_lib() is not None

    def _read_blosc_pixel_rows(self, rows: np.ndarray) -> np.ndarray:
        pixels = self.h5_file['pixels']
        chunk_frames = int(pixels.chunks[0])
        output = np.empty((*rows.shape, *pixels.shape[1:]), dtype=pixels.dtype)
        locations: dict[int, list[tuple[int, int, int]]] = {}
        for batch_idx, sample_rows in enumerate(rows):
            for time_idx, row in enumerate(sample_rows):
                row = int(row)
                locations.setdefault(row // chunk_frames, []).append((batch_idx, time_idx, row % chunk_frames))
        chunk_shape = (chunk_frames, *pixels.shape[1:])
        chunk_nbytes = int(np.prod(chunk_shape)) * pixels.dtype.itemsize
        lib = self._get_blosc_lib()
        for chunk_idx, targets in locations.items():
            filter_mask, raw = pixels.id.read_direct_chunk((chunk_idx * chunk_frames, 0, 0, 0))
            if filter_mask & 1:
                chunk = np.frombuffer(raw, dtype=pixels.dtype).reshape(chunk_shape)
            else:
                chunk = np.empty(chunk_shape, dtype=pixels.dtype)
                written = lib.blosc_decompress(ctypes.cast(ctypes.c_char_p(raw), ctypes.c_void_p), chunk.ctypes.data_as(ctypes.c_void_p), chunk_nbytes)
                if written != chunk_nbytes:
                    raise OSError(f'Blosc direct-chunk decompression returned {written} bytes, expected {chunk_nbytes}')
            for batch_idx, time_idx, row_in_chunk in targets:
                output[batch_idx, time_idx] = chunk[row_in_chunk]
        return output

    def __getitems__(self, indices: list[int]) -> list[dict]:
        if not indices or not self._can_direct_read_blosc_pixels():
            return [self[idx] for idx in indices]
        clips = [self.clip_indices[int(idx)] for idx in indices]
        starts = np.asarray([self.offsets[ep_idx] + start for ep_idx, start in clips], dtype=np.int64)
        observation_rows = starts[:, None] + np.arange(self.num_steps, dtype=np.int64)[None, :] * self.frameskip
        pixel_batch = self._read_blosc_pixel_rows(observation_rows)
        batch = []
        for batch_idx, (ep_idx, start) in enumerate(clips):
            end = start + self.span
            g_start, g_end = (self.offsets[ep_idx] + start, self.offsets[ep_idx] + end)
            steps = {}
            for col in self._keys:
                if col == 'pixels':
                    data = pixel_batch[batch_idx]
                else:
                    src = self._cache if col in self._cache else self.h5_file
                    data = self._read_column(src[col], col, ep_idx, g_start, g_end)
                if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                    val = data[0] if len(data) > 0 else b''
                    steps[col] = val.decode() if isinstance(val, bytes) else val
                else:
                    steps[col] = torch.from_numpy(data)
                    if data.ndim == 4 and data.shape[-1] in (1, 3):
                        steps[col] = steps[col].permute(0, 3, 1, 2)
            if self.transform:
                steps = self.transform(steps)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(self.num_steps, -1)
            batch.append(steps)
        return batch

    def _read_column(self, source, col, ep_idx, g_start, g_end):
        stop = min(g_end, int(self.offsets[ep_idx] + self.lengths[ep_idx])) if self.last_action_unused else g_end
        selection = slice(g_start, stop) if col == 'action' else slice(g_start, stop, self.frameskip)
        data = source[selection]
        if col == 'action' and stop < g_end:
            padding = [(0, g_end - stop)] + [(0, 0)] * (data.ndim - 1)
            data = np.pad(data, padding)
        return data

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start, g_end = (self.offsets[ep_idx] + start, self.offsets[ep_idx] + end)
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = self._read_column(src[col], col, ep_idx, g_start, g_end)
            if data.dtype == np.object_ or data.dtype.kind in ('S', 'U'):
                val = data[0] if len(data) > 0 else b''
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)
        return self.transform(steps) if self.transform else steps

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self.h5_file[col][:]

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        return {col: self.h5_file[col][row_idx] for col in self._keys}

    def merge_col(self, source: list[str] | str, target: str, dim: int=-1) -> None:
        self._open()
        if isinstance(source, str):
            source = [k for k in self.h5_file.keys() if re.match(source, k)]
        merged = np.concatenate([self._get_col(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(f"Merged columns {source} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1

class HDF5Writer:

    def __init__(self, path, *, mode: str='append', compression: str | None=None, compression_opts=None, libver='latest'):
        validate_write_mode(mode)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.compression = compression
        self.compression_opts = compression_opts
        self.libver = libver
        self._f: h5py.File | None = None
        self._initialized = False
        self._appending_existing = False
        self._ep_written = 0
        self._global_ptr = 0

    def __enter__(self):
        exists = self.path.exists()
        if exists and self.mode == 'error':
            raise FileExistsError(f"HDF5Writer: '{self.path}' already exists. Pass mode='overwrite' to replace it or mode='append' to extend it.")
        if self.mode == 'overwrite' or not exists:
            self._f = h5py.File(str(self.path), 'w', libver=self.libver)
        else:
            self._f = h5py.File(str(self.path), 'a', libver=self.libver)
            self._load_existing_state()
        return self

    def __exit__(self, *exc):
        if self._f is not None:
            self._f.close()
            self._f = None

    def write_episode(self, ep_data: dict) -> None:
        if self._f is None:
            raise RuntimeError('HDF5Writer used outside of a `with` block')
        if not self._initialized:
            self._init_schema(ep_data)
            self._initialized = True
        elif self._appending_existing and self._ep_written == 0:
            self._validate_episode_against_existing(ep_data)
        ep_len = len(next(iter(ep_data.values())))
        for col, vals in ep_data.items():
            ds = self._f[col]
            ds.resize(self._global_ptr + ep_len, axis=0)
            ds[self._global_ptr:self._global_ptr + ep_len] = np.array(vals)
        n = self._f['ep_len'].shape[0]
        self._f['ep_len'].resize(n + 1, axis=0)
        self._f['ep_len'][n] = ep_len
        self._f['ep_offset'].resize(n + 1, axis=0)
        self._f['ep_offset'][n] = self._global_ptr
        self._ep_written += 1
        self._global_ptr += ep_len

    def write_episodes(self, episodes) -> None:
        for ep in episodes:
            self.write_episode(ep)

    def _load_existing_state(self) -> None:
        if 'ep_len' not in self._f or 'ep_offset' not in self._f:
            raise ValueError(f"HDF5Writer: cannot append to '{self.path}' — file is missing ep_len/ep_offset metadata.")
        ep_len = self._f['ep_len'][:]
        self._global_ptr = int(ep_len.sum()) if len(ep_len) else 0
        self._initialized = True
        self._appending_existing = True

    def _validate_episode_against_existing(self, ep_data: dict) -> None:
        existing = {k for k in self._f.keys() if k not in ('ep_len', 'ep_offset')}
        incoming = set(ep_data)
        missing = existing - incoming
        extra = incoming - existing
        if missing or extra:
            raise ValueError(f"HDF5Writer: append failed — schema mismatch on '{self.path}'. Missing columns: {sorted(missing)}; unexpected columns: {sorted(extra)}.")
        for col, vals in ep_data.items():
            sample = np.asarray(vals[0])
            ds_shape = self._f[col].shape[1:]
            if sample.shape != ds_shape:
                raise ValueError(f"HDF5Writer: append failed — column '{col}' shape mismatch: existing per-step={ds_shape}, incoming per-step={sample.shape}.")

    def _init_schema(self, sample_ep: dict) -> None:
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            compression_kwargs = {}
            if self.compression is not None:
                compression_kwargs['compression'] = self.compression
                if self.compression_opts is not None:
                    compression_kwargs['compression_opts'] = self.compression_opts
            self._f.create_dataset(col, shape=(0, *sample.shape), maxshape=(None, *sample.shape), dtype=sample.dtype, chunks=(1, *sample.shape), **compression_kwargs)
        self._f.create_dataset('ep_len', shape=(0,), maxshape=(None,), dtype=np.int32)
        self._f.create_dataset('ep_offset', shape=(0,), maxshape=(None,), dtype=np.int64)

@register_format
class HDF5(Format):
    name = 'hdf5'

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        if p.suffix in ('.h5', '.hdf5'):
            return True
        if p.is_dir():
            return any(p.glob('*.h5')) or any(p.glob('*.hdf5'))
        return False

    @classmethod
    def open_reader(cls, path, **kwargs) -> HDF5Dataset:
        p = Path(path)
        if p.is_dir():
            files = sorted(p.glob('*.h5')) + sorted(p.glob('*.hdf5'))
            if not files:
                raise FileNotFoundError(f'No .h5/.hdf5 file in {p}')
            if len(files) > 1:
                raise ValueError(f'Ambiguous dataset: multiple HDF5 files in {p}. Pass the file directly.')
            p = files[0]
        return HDF5Dataset(path=p, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> HDF5Writer:
        return HDF5Writer(path, **kwargs)
__all__ = ['HDF5', 'HDF5Dataset', 'HDF5Writer']
