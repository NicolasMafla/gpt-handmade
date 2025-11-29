import random
import numpy as np
from functools import reduce
from itertools import chain
from abc import ABC, abstractmethod
from typing import List
from .engine import Value, Tensor

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

class TensorModule(ABC):
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    @abstractmethod
    def parameters(self) -> List[Tensor]:
        pass

class TensorNeuron(TensorModule):
    def __init__(self, nin: int, nonlin: bool=True):
        self.w = Tensor(data=np.random.uniform(low=-1.0, high=1.0, size=nin))
        self.b = Tensor(data=np.zeros(nin))
        self.nonlin = nonlin

    def __call__(self, x: Tensor) -> Tensor:
        act = (self.w * x + self.b).sum()
        return act.relu() if self.nonlin else act

    def parameters(self) -> List[Tensor]:
        return [self.w] + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'}TensorNeuron({len(self.w.data)})"

class TensorLayer(TensorModule):
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [TensorNeuron(nin=nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: Tensor) -> Tensor:
        outs = [n(x) for n in self.neurons]
        out = reduce(lambda x, y: x.concat(y), outs)
        return out

    def parameters(self) -> List[Tensor]:
        l_params = [n.parameters() for n in self.neurons]
        params = list(chain.from_iterable(l_params))
        return params

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class TensorMLP(TensorModule):
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [TensorLayer(nin=sz[i], nout=sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        l_params = [n.parameters() for n in self.layers]
        params = list(chain.from_iterable(l_params))
        return params

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"