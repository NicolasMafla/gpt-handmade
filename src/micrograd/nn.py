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

class TensorLayer(TensorModule):
    def __init__(self, nin: int, nout: int, activation: str='Linear'):
        self.w = Tensor(data=np.random.uniform(low=-1.0, high=1.0, size=(nin, nout)))
        self.b = Tensor(data=np.zeros(nout))
        self.activation = activation

    def __call__(self, x: Tensor) -> Tensor:
        z = x.matmul(self.w) + self.b
        if self.activation == 'Relu':
            return z.relu()
        elif self.activation == 'Tanh':
            return z.tanh()
        else:
            return z

    def parameters(self) -> List[Tensor]:
        return [self.w, self.b]

    def __repr__(self) -> str:
        return f"{self.activation}Layer({self.w.data.shape[0]} -> {self.w.data.shape[1]})"

class TensorMLP(TensorModule):
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [TensorLayer(nin=sz[i], nout=sz[i+1], activation='Relu') for i in range(len(nouts))]
        self.layers[-1].activation = 'Linear'

    @classmethod
    def from_layers(cls, layers: list[TensorLayer]):
        obj = cls.__new__(cls)
        obj.layers = layers[:]
        return obj

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