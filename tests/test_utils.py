import torch

from torch_einops_utils.torch_einops_utils import (
    exists,
    pad_at_dim,
    pad_left_at_dim,
    pad_right_at_dim,
    pack_with_inverse
)

def test_exist():
    assert not exists(None)

def test_pad_at_dim():
    t = torch.randn(3, 6, 1)
    padded = pad_at_dim(t, (0, 1), dim = 1)

    assert padded.shape == (3, 7, 1)
    assert torch.allclose(padded, pad_right_at_dim(t, 1, dim = 1))
    assert not torch.allclose(padded, pad_left_at_dim(t, 1, dim = 1))

def test_pack_with_inverse():
    t = torch.randn(3, 12, 2, 2)
    t, inverse = pack_with_inverse(t, 'b * d')

    assert t.shape == (3, 24, 2)
    t = inverse(t)
    assert t.shape == (3, 12, 2, 2)

    u = torch.randn(3, 4, 2)
    t, inverse = pack_with_inverse([t, u], 'b * d')
    assert t.shape == (3, 28, 2)
    t, u = inverse(t)
    assert t.shape == (3, 12, 2, 2)
    assert u.shape == (3, 4, 2)
