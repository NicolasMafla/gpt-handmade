import torch
from src.micrograd import Value

def test_operations():
    a = Value(-4.0)
    b = Value(3.0)

    assert (a + b).data == -1.0
    assert (b + a).data == -1.0
    assert (a ** 2).data == 16.0
    assert (-a).data == 4.0
    assert (a.relu()).data == 0.0
    assert (b.relu()).data == 3.0

def test_backward():
    a = Value(4.0)
    b = Value(3.0)
    c = a * b
    d = c.relu()

    d.backward()

    assert d.grad == 1.0
    assert c.grad == 1.0
    assert b.grad == 4.0
    assert a.grad == 3.0

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    z = y.exp()
    z.backward()
    xmg, zmg = x, z

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).tanh()
    y = h + q + q * x
    z = y.exp()
    z.backward()
    xpt, zpt = x, z

    assert zmg.data == zpt.data.item()
    assert xmg.grad == xpt.grad.item()

def test_deep_sanity_check():
    tol = 1e-6

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    assert abs(gmg.data - gpt.data.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol