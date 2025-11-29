import torch
import random
import numpy as np
from functools import reduce
from src.micrograd import Tensor, TensorNeuron, TensorLayer, TensorMLP

def test_operations():
    xmg = Tensor([-4.0, 3.0, 6.0])
    wmg = Tensor([2.0, -1.0, -6.0])
    cmg = wmg / 5 - xmg ** 2 + 10
    amg = cmg.relu()
    Lmg = amg.sum()
    Lmg.backward()

    xpt = torch.Tensor([-4.0, 3.0, 6.0]).double()
    wpt = torch.Tensor([2.0, -1.0, -6.0]).double()
    wpt.requires_grad = True
    cpt = wpt / 5 - xpt ** 2 + 10
    apt = cpt.relu()
    Lpt = apt.sum()
    Lpt.backward()

    assert Lmg.data.item() == Lpt.data.item()
    assert np.allclose(wmg.grad, wpt.grad.numpy(), atol=1e-6)

def test_nn():
    np.random.seed(0)

    epochs = 50
    lr = 0.001

    n = TensorMLP(nin=3, nouts=[4, 5, 1])

    xs = [
        Tensor([2.0, 3.0, -1.0]),
        Tensor([3.0, -1.0, 0.5]),
        Tensor([0.5, 1.0, 1.0]),
        Tensor([1.0, 1.0, -1.0]),
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    for k in range(epochs):
        ypred = [n(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        n.zero_grad()
        loss.backward()

        for p in n.parameters():
            p.data += -lr * p.grad

        print(loss.data.item())

    print(ypred)
    assert loss.data.item() < 5
