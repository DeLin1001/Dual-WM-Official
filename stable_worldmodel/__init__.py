import importlib
__all__ = ['World', 'PlanConfig', 'data', 'envs', 'policy', 'solver', 'spaces', 'utils', 'wm', 'wrapper']
_LAZY_SUBMODULES = {'data', 'envs', 'policy', 'solver', 'spaces', 'utils', 'wm', 'wrapper'}
_LAZY_ATTRS = {'World': ('stable_worldmodel.world', 'World'), 'PlanConfig': ('stable_worldmodel.policy', 'PlanConfig')}

def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f'stable_worldmodel.{name}')
        globals()[name] = mod
        return mod
    if name in _LAZY_ATTRS:
        (modpath, attrname) = _LAZY_ATTRS[name]
        mod = importlib.import_module(modpath)
        attr = getattr(mod, attrname)
        globals()[name] = attr
        if name == 'World':
            importlib.import_module('stable_worldmodel.envs')
        return attr
    raise AttributeError(f"module 'stable_worldmodel' has no attribute {name!r}")

def __dir__():
    return sorted(set(__all__) | set(globals().keys()))
