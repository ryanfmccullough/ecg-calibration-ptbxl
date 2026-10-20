import pytest

torch = pytest.importorskip("torch")

from ecgcal.model import ECGResNet


def test_forward_shape():
    m = ECGResNet(n_leads=12, n_classes=5, width=8)
    x = torch.randn(4, 12, 1000)
    assert m(x).shape == (4, 5)
