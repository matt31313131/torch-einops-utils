from __future__ import annotations

import torch
from torch import is_tensor
import torch.nn.functional as F

from einops import rearrange, repeat, reduce, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(arr):
    return arr[0]

# exported functions

# padding

def pad_at_dim(
    t,
    pad: tuple[int, int],
    *,
    dim = -1,
    value = 0.
):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_left_at_dim(t, pad: int, **kwargs):
    return pad_at_dim(t, (pad, 0), **kwargs)

def pad_right_at_dim(t, pad: int, **kwargs):
    return pad_at_dim(t, (0, pad), **kwargs)

# einops pack

def pack_with_inverse(t, pattern):
    is_one = is_tensor(t)

    if is_one:
        t = [t]

    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out = unpack(out, packed_shape, inv_pattern)

        if is_one:
            out = first(out)

        return out

    return packed, inverse
