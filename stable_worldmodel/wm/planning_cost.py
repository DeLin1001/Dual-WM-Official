import math
import torch
import torch.nn.functional as F
from .utils import MSE_Cost as MSE

def _goal_vector(goal_emb: torch.Tensor) -> torch.Tensor:
    if goal_emb.ndim == 2:
        return goal_emb
    if goal_emb.ndim == 3:
        return goal_emb[:, -1]
    if goal_emb.ndim == 4:
        return goal_emb[:, :, -1]
    raise ValueError(f'Unsupported goal_emb shape: {tuple(goal_emb.shape)}')

def future_predictions(info_dict: dict) -> torch.Tensor:
    pred_emb = info_dict['predicted_emb']
    history_len = info_dict['pixels'].shape[2] if 'pixels' in info_dict and torch.is_tensor(info_dict['pixels']) else 0
    if history_len >= pred_emb.shape[2]:
        return pred_emb[:, :, -1:]
    return pred_emb[:, :, history_len:]

    
    
def goal_planning_cost(model, info_dict: dict) -> torch.Tensor:
    pred_future = future_predictions(info_dict)
    goal = _goal_vector(info_dict['goal_emb'])
    if goal.ndim == 2:
        goal = goal[:, None, None, :]
    elif goal.ndim == 3:
        goal = goal[:, :, None, :]
    return MSE(model, pred_future, goal)




__all__ = ['future_predictions',  'goal_planning_cost']
