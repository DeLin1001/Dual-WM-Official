import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import numpy as np
from loguru import logger as logging

def exists(val: Any) -> bool:
    return val is not None

def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d

def flatten_dict(d: dict, parent_key: str='', sep: str='.') -> dict:
    items: dict = {}
    for (k, v) in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def get_in(mapping: Any, path: Iterable[str]) -> Any:
    cur = mapping
    for key in list(path):
        cur = cur[key]
    return cur

DEFAULT_CACHE_DIR = '.'
