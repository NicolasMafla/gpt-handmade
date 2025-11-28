import torch
import random
import numpy as np
from src.micrograd import Tensor, MLP

def test_operations():
    xmg = Tensor([-4.0, 3.0, 6.0])
    wmg = Tensor([2.0, -1.0, -6.0])
    cmg = wmg / 5 - xmg ** 2
    amg = cmg.relu()
    Lmg = amg.sum()
    Lmg.backward()

    xpt = torch.Tensor([-4.0, 3.0, 6.0]).double()
    wpt = torch.Tensor([2.0, -1.0, -6.0]).double()
    wpt.requires_grad = True
    cpt = wpt / 5 - xpt ** 2
    apt = cpt.relu()
    Lpt = apt.sum()
    Lpt.backward()

    assert Lmg.data.item() == Lpt.data.item()
    assert np.allclose(wmg.grad, wpt.grad.numpy(), atol=1e-6)
