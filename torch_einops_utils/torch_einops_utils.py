from __future__ import annotations

import torch
from torch.nn.functional import pad

from einops import rearrange, repeat, reduce, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# exported functions
