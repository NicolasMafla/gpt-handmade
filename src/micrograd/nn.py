import random
from abc import ABC, abstractmethod
from typing import List
from .engine import Value

class Module(ABC):
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    @abstractmethod
    def parameters(self) -> List[Value]:
        pass

class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool=True):
        self.w = [Value(data=random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(data=0)
        self.nonlin = nonlin

    def __call__(self, x: List[float]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin=nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: List[float]) -> List[Value] | Value:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(nin=sz[i], nout=sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x: List[float]) -> List[Value] | Value:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"