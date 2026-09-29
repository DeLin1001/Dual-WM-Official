import math

import torch
from torch.utils.data import Subset


class EpisodeSubset(Subset):

    def __init__(self, dataset, episode_ids, indices):
        super().__init__(dataset, indices)
        self.episode_ids = tuple(sorted(episode_ids))


def split_by_episode(dataset, train_fraction=0.9, seed=42):
    fraction = float(train_fraction)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('The training episode fraction must be between zero and one')
    episode_count = len(dataset.lengths)
    train_count = int(episode_count * fraction)
    if not 0 < train_count < episode_count:
        raise ValueError('Episode splitting requires nonempty training and held-out episode groups')
    generator = torch.Generator().manual_seed(int(seed))
    episodes = torch.randperm(episode_count, generator=generator).tolist()
    train_episodes = episodes[:train_count]
    test_episodes = episodes[train_count:]
    train_owners = set(train_episodes)
    train_indices = []
    test_indices = []
    for index, (episode, _) in enumerate(dataset.clip_indices):
        if episode in train_owners:
            train_indices.append(index)
        else:
            test_indices.append(index)
    if not train_indices or not test_indices:
        raise ValueError('The episode split has an empty window group; use longer episodes or a shorter rollout')
    return EpisodeSubset(dataset, train_episodes, train_indices), EpisodeSubset(dataset, test_episodes, test_indices)
