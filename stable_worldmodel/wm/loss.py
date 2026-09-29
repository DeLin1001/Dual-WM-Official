import math
import torch
import torch.distributed as dist
import torch.nn.functional as F

class SIGReg(torch.nn.Module):

    def __init__(self, num_projections=256, distributed=False):
        super().__init__()
        self.K = num_projections
        self.distributed = distributed
        self._cached_B = -1
        self._cached_target = None

    @staticmethod
    def _distributed_is_ready():
        return dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)

    def _gather_batch_with_grad(self, proj):
        if not self.distributed or not self._distributed_is_ready():
            return proj
        from torch.distributed.nn.functional import all_gather
        gathered = all_gather(proj)
        return torch.cat(gathered, dim=1)

    def _get_target(self, B, device, dtype):
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device, dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device, dtype=dtype)

    def forward(self, proj):
        proj = self._gather_batch_with_grad(proj)
        (_, B, D) = proj.shape
        mu = proj.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()
        z_centered = proj - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)) + 1e-06
        scale_loss = (std - 1.0).pow(2).mean()
        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(torch.randn(D, self.K, device=proj.device, dtype=proj.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, proj.device, proj.dtype).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()
        return scale_loss + shape_loss + center_loss
