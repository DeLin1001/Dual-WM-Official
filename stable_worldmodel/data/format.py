from __future__ import annotations
from collections.abc import Iterable
from typing import Protocol, runtime_checkable
FORMATS: dict[str, type[Format]] = {}
WRITE_MODES = ('append', 'overwrite', 'error')

def validate_write_mode(mode: str) -> str:
    if mode not in WRITE_MODES:
        raise ValueError(f'write mode must be one of {WRITE_MODES}, got {mode!r}')
    return mode

def register_format(cls: type[Format]) -> type[Format]:
    name = getattr(cls, 'name', None)
    if not name:
        raise ValueError(f'{cls.__name__} must set a non-empty `name` class attribute')
    if name in FORMATS:
        raise ValueError(f'format {name!r} is already registered')
    FORMATS[name] = cls
    return cls

def list_formats() -> list[str]:
    return list(FORMATS)

def get_format(name: str) -> type[Format]:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(f'unknown format {name!r}; available: {list_formats()}') from None

def detect_format(path) -> type[Format] | None:
    for fmt in FORMATS.values():
        if fmt.detect(path):
            return fmt
    return None

class Format:
    name: str = ''

    @classmethod
    def detect(cls, path) -> bool:
        raise NotImplementedError(f'{cls.__name__}.detect must be implemented')

    @classmethod
    def open_reader(cls, path, **kwargs):
        raise NotImplementedError(f'format {cls.name or cls.__name__!r} does not support reading')

    @classmethod
    def open_writer(cls, path, **kwargs) -> Writer:
        raise NotImplementedError(f'format {cls.name or cls.__name__!r} does not support writing (read-only)')

@runtime_checkable
class Writer(Protocol):

    def __enter__(self) -> Writer:
        ...

    def __exit__(self, *exc) -> None:
        ...

    def write_episode(self, ep_data: dict) -> None:
        ...

    def write_episodes(self, episodes: Iterable[dict]) -> None:
        ...
__all__ = ['FORMATS', 'Format', 'Writer', 'detect_format', 'get_format', 'list_formats', 'register_format']
