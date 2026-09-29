import os
from pathlib import Path
from stable_worldmodel.utils import DEFAULT_CACHE_DIR
from stable_worldmodel.data.normalization import IdentityScaler, PercentileScaler, ZScoreScaler, get_scaler
import numpy as np

def get_cache_dir(override_root: Path | None=None, sub_folder: str | None=None) -> Path:
    base = override_root
    if override_root is None:
        base = os.getenv('STABLEWM_HOME', str(DEFAULT_CACHE_DIR))
    cache_path = Path(base, sub_folder) if sub_folder is not None else Path(base)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path

def ensure_dir_exists(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def column_normalizer(dataset, source: str, target: str, method: str='zscore'):
    from stable_pretraining.data.transforms import WrapTorchTransform
    scaler = get_scaler(method)
    if method != 'none':
        data = np.array(dataset.get_col_data(source))
        scaler.fit(data)
    return WrapTorchTransform(scaler, source=source, target=target)

def load_dataset(name, cache_dir=None, format=None, **kwargs):
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    if format not in (None, 'hdf5'):
        raise ValueError('This release supports HDF5 datasets only')
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = get_cache_dir(cache_dir, 'datasets') / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return HDF5Dataset(path=path, **kwargs)
