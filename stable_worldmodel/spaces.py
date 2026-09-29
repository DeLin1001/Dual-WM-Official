import time
from typing import Any
from collections.abc import Callable, Generator, Iterable, Sequence
from gymnasium import spaces
from loguru import logger as logging
import stable_worldmodel as swm

def reset_variation_space(variation_space: spaces.Space, seed: int | None=None, options: dict | None=None, default_variations: set | None=None) -> None:
    variation_space.seed(seed)
    variation_space.reset()
    options = options or {}
    var_keys = options.get('variation', default_variations or ())
    if not isinstance(var_keys, Sequence):
        raise ValueError('variation option must be a Sequence containing variation names')
    variation_space.update(var_keys)
    if 'variation_values' in options:
        variation_space.set_value(options['variation_values'])
    assert variation_space.check(debug=True), 'Variation values must be within variation space!'
    return

class Discrete(spaces.Discrete):

    def __init__(self, n: int, init_value: int | None=None, constrain_fn: Callable[[int], bool] | None=None, **kwargs: Any) -> None:
        super().__init__(n, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self) -> int | None:
        return self._init_value

    @property
    def value(self) -> int | None:
        return self._value

    def reset(self) -> None:
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        return super().contains(x) and self.constrain_fn(x)

    def check(self) -> bool:
        if not self.constrain_fn(self.value):
            logging.warning(f'Discrete: value {self.value} does not satisfy constrain_fn')
            return False
        return super().contains(self.value)

    def sample(self, mask: Any | None=None, max_tries: int=1000, warn_after_s: float | None=5.0, set_value: bool=True, **kwargs: Any) -> int:
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(mask=mask)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and time.time() - start > warn_after_s:
                logging.warning('rejection sampling: rejection sampling is taking a while...')
        raise RuntimeError(f'rejection sampling: predicate not satisfied after {max_tries} draws')

    def set_init_value(self, value: int) -> None:
        if not self.contains(value):
            raise ValueError(f'Value {value} is not contained in the Discrete space')
        self._init_value = value

    def set_value(self, value: int) -> None:
        if not self.contains(value):
            raise ValueError(f'Value {value} is not contained in the Discrete space')
        self._value = value

class MultiDiscrete(spaces.MultiDiscrete):

    def __init__(self, nvec: Any, init_value: Any | None=None, constrain_fn: Callable[[Any], bool] | None=None, **kwargs: Any) -> None:
        super().__init__(nvec, **kwargs)
        self._init_value = init_value
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._value = init_value

    @property
    def init_value(self) -> Any | None:
        return self._init_value

    @property
    def value(self) -> Any | None:
        return self._value

    def reset(self) -> None:
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        return super().contains(x) and self.constrain_fn(x)

    def check(self) -> bool:
        if not self.constrain_fn(self.value):
            logging.warning(f'MultiDiscrete: value {self.value} does not satisfy constrain_fn')
            return False
        return super().contains(self.value)

    def sample(self, mask: Any | None=None, max_tries: int=1000, warn_after_s: float | None=5.0, set_value: bool=True, **kwargs: Any) -> Any:
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(mask=mask)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and time.time() - start > warn_after_s:
                logging.warning('rejection sampling: rejection sampling is taking a while...')
        raise RuntimeError(f'rejection sampling: predicate not satisfied after {max_tries} draws')

    def set_init_value(self, value: Any) -> None:
        if not self.contains(value):
            raise ValueError(f'Value {value} is not contained in the MultiDiscrete space')
        self._init_value = value

    def set_value(self, value: Any) -> None:
        if not self.contains(value):
            raise ValueError(f'Value {value} is not contained in the MultiDiscrete space')
        self._value = value

class Box(spaces.Box):

    def __init__(self, low: Any, high: Any, shape: Iterable[int] | None=None, init_value: Any | None=None, constrain_fn: Callable[[Any], bool] | None=None, **kwargs: Any) -> None:
        super().__init__(low, high, shape, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._init_value = init_value
        self._value = init_value

    @property
    def init_value(self) -> Any | None:
        return self._init_value

    @property
    def value(self) -> Any | None:
        return self._value

    def reset(self) -> None:
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        return super().contains(x) and self.constrain_fn(x)

    def check(self) -> bool:
        if not self.constrain_fn(self.value):
            logging.warning(f'Box: value {self.value} does not satisfy constrain_fn')
            return False
        return self.contains(self.value)

    def sample(self, mask: Any | None=None, max_tries: int=1000, warn_after_s: float | None=5.0, set_value: bool=True, **kwargs: Any) -> Any:
        start = time.time()
        for i in range(max_tries):
            sample = super().sample(mask=mask)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and time.time() - start > warn_after_s:
                logging.warning('rejection sampling: rejection sampling is taking a while...')
        raise RuntimeError(f'rejection sampling: predicate not satisfied after {max_tries} draws')

    def set_init_value(self, value: Any) -> None:
        if not self.contains(value):
            raise ValueError(f'Value {value} is not contained in the Box space')
        self._init_value = value

    def set_value(self, value: Any) -> None:
        if not self.contains(value):
            raise ValueError(f'Value {value} is not contained in the Box space')
        self._value = value

class RGBBox(Box):

    def __init__(self, shape: Iterable[int]=(3,), init_value: Any | None=None, **kwargs: Any) -> None:
        if not any((dim == 3 for dim in shape)):
            raise ValueError('shape must have a channel of size 3')
        super().__init__(low=0, high=255, shape=shape, dtype='uint8', init_value=init_value, **kwargs)

class Dict(spaces.Dict):

    def __init__(self, spaces_dict: dict[Any, spaces.Space] | None=None, init_value: dict | None=None, constrain_fn: Callable[[dict], bool] | None=None, sampling_order: list[str] | None=None, **kwargs: Any) -> None:
        super().__init__(spaces_dict, **kwargs)
        self.constrain_fn = constrain_fn or (lambda x: True)
        self._init_value = init_value
        self._value = self.init_value
        if sampling_order is None:
            self._sampling_order = list(self.spaces.keys())
        elif len(sampling_order) != len(self.spaces):
            missing_keys = set(self.spaces.keys()).difference(set(sampling_order))
            logging.warning(f'Dict sampling_order is missing keys {missing_keys}, adding them at the end of the sampling order')
            self._sampling_order = list(sampling_order) + [str(k) for k in missing_keys]
        else:
            self._sampling_order = sampling_order
        if not all((key in self.spaces for key in self._sampling_order)):
            missing = set(self._sampling_order) - set(self.spaces.keys())
            raise ValueError(f'sampling_order contains keys not in spaces: {missing}')

    @property
    def init_value(self) -> dict:
        init_val: dict = {}
        for (k, v) in self.spaces.items():
            if hasattr(v, 'init_value'):
                init_val[k] = v.init_value
            else:
                logging.warning(f'Space {k} of type {type(v)} does not have init_value property, using default sample instead')
                init_val[k] = v.sample()
        return init_val

    @property
    def value(self) -> dict:
        val: dict = {}
        for (k, v) in self.spaces.items():
            if hasattr(v, 'value'):
                val[k] = v.value
            else:
                raise ValueError(f'Space {k} of type {type(v)} does not have value property')
        return val

    def _get_sampling_order(self, parts: tuple[str, ...] | None=None) -> Generator[str, None, None]:
        if parts is None:
            parts = ()
        keys = getattr(self, '_sampling_order', None) or self.spaces.keys()
        for key in keys:
            if key not in self.spaces:
                continue
            key_str = str(key)
            path = parts + (key_str,)
            yield '.'.join(path)
            subspace = self.spaces[key]
            if isinstance(subspace, spaces.Dict):
                if hasattr(subspace, '_get_sampling_order'):
                    yield from subspace._get_sampling_order(path)
                else:
                    for subkey in subspace.spaces.keys():
                        yield '.'.join(path + (str(subkey),))

    @property
    def sampling_order(self) -> list[str]:
        return list(self._get_sampling_order())

    def reset(self) -> None:
        for v in self.spaces.values():
            if hasattr(v, 'reset'):
                v.reset()
        self._value = self.init_value

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        for key in self.spaces.keys():
            if key not in x:
                return False
            if not self.spaces[key].contains(x[key]):
                return False
        if not self.constrain_fn(x):
            return False
        return True

    def check(self, debug: bool=False) -> bool:
        for (k, v) in self.spaces.items():
            if hasattr(v, 'check'):
                if not v.check():
                    if debug:
                        logging.warning(f'Dict: space {k} failed check()')
                    return False
        return True

    def names(self) -> list[str]:

        def _key_generator(d: dict[Any, spaces.Space], parent_key: str='') -> Generator[str, None, None]:
            for (k, v) in d.items():
                new_key = f'{parent_key}.{k}' if parent_key else k
                if isinstance(v, spaces.Dict):
                    yield from _key_generator(v.spaces, new_key)
                else:
                    yield new_key
        return list(_key_generator(self.spaces))

    def sample(self, mask: Any | None=None, max_tries: int=1000, warn_after_s: float | None=5.0, set_value: bool=True, **kwargs: Any) -> dict:
        start = time.time()
        for i in range(max_tries):
            sample: dict = {}
            for k in self._sampling_order:
                sub_mask = mask[k] if isinstance(mask, dict) and k in mask else None
                sample[k] = self.spaces[k].sample(mask=sub_mask, set_value=set_value, **kwargs)
            if self.contains(sample):
                if set_value:
                    self._value = sample
                return sample
            if warn_after_s is not None and time.time() - start > warn_after_s:
                logging.warning('rejection sampling is taking a while...')
        raise RuntimeError(f'constrain_fn not satisfied after {max_tries} draws')

    def update(self, keys: Iterable[str]) -> None:
        keys_set = set(keys)
        order = self.sampling_order
        if len(keys_set) == 1 and 'all' in keys_set:
            self.sample()
        else:
            for v in filter(keys_set.__contains__, order):
                try:
                    var_path = v.split('.')
                    swm.utils.get_in(self, var_path).sample()
                except (KeyError, TypeError):
                    raise ValueError(f'Key {v} not found in Dict space')
        assert self.check(debug=True), 'Values must be within space!'

    def set_init_value(self, variations_values: dict) -> None:
        for (k, v) in variations_values.items():
            try:
                var_path = k.split('.')
                space = swm.utils.get_in(self, var_path)
                assert space.contains(v), f'Value {v} for key {k} is not contained in the space'
                space.set_init_value(v)
            except (KeyError, TypeError):
                raise ValueError(f'Key {k} not found in Dict space')

    def set_value(self, variations_values: dict) -> None:
        for (k, v) in variations_values.items():
            try:
                var_path = k.split('.')
                space = swm.utils.get_in(self, var_path)
                assert space.contains(v), f'Value {v} for key {k} is not contained in the space'
                space.set_value(v)
            except (KeyError, TypeError):
                raise ValueError(f'Key {k} not found in Dict space')

    def to_str(self) -> str:

        def _tree(d: dict[Any, spaces.Space], indent: int=0) -> str:
            lines = []
            for (k, v) in d.items():
                if isinstance(v, dict | self.__class__ | spaces.Dict):
                    lines.append('    ' * indent + f'{k}:')
                    sub_dict = v.spaces if isinstance(v, spaces.Dict) else v
                    lines.append(_tree(sub_dict, indent + 1))
                else:
                    lines.append('    ' * indent + f'{k}: {v}')
            return '\n'.join(lines)
        return _tree(self.spaces)
