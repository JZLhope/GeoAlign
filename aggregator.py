from typing import Callable, Optional
import torch
import torch.nn.functional as F

def _l2n(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(v, dim=dim)

def avg_pool(feat: torch.Tensor) -> torch.Tensor:
    vec = feat.mean(dim=(1, 2))
    return _l2n(vec, dim=0)

def max_pool(feat: torch.Tensor) -> torch.Tensor:
    vec, _ = feat.max(dim=1)
    vec, _ = vec.max(dim=1)
    return _l2n(vec, dim=0)

def gem_pool(feat: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(feat, min=eps)
    x = x.pow(p).mean(dim=(1, 2)).pow(1.0 / p)
    return _l2n(x, dim=0)

def get_aggregator(name: str, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:

    name = str(name).lower()
    if name == 'avg':
        return avg_pool
    elif name == 'max':
        return max_pool
    elif name == 'gem':
        p = kwargs.get('p', 3.0)
        return lambda feat: gem_pool(feat, p=p)
    else:
        raise ValueError(f"known: {name}")