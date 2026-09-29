import gymnasium as gym
import numpy as np

def is_discrete_space(space: gym.Space) -> bool:
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return True
    return isinstance(space, gym.spaces.Box) and np.issubdtype(space.dtype, np.integer)
__all__ = ['is_discrete_space']
