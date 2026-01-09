import torch

from torch_einops_utils.torch_einops_utils import (
    exists,
    pad_at_dim,
    pad_left_at_dim,
    pad_right_at_dim
)

def test_exist():
    assert not exists(None)

def test_pad_at_dim():
    t = torch.randn(3, 6, 1)
    padded = pad_at_dim(t, (0, 1), dim = 1)

    assert padded.shape == (3, 7, 1)
    assert torch.allclose(padded, pad_right_at_dim(t, 1, dim = 1))
    assert not torch.allclose(padded, pad_left_at_dim(t, 1, dim = 1))
