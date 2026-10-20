import pytest

torch = pytest.importorskip("torch")

from ecgcal.train import default_device


def test_default_device_prefers_mps_then_cuda_then_cpu():
    dev = default_device()
    if torch.backends.mps.is_available():
        assert dev == "mps"
    elif torch.cuda.is_available():
        assert dev == "cuda"
    else:
        assert dev == "cpu"
    torch.device(dev)  # a valid device string
