
from torch_einops_utils.torch_einops_utils import exists

def test_exist():
    assert not exists(None)
