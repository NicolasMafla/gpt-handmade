import math

class Value:

    def __init__(self, data: int | float, _children: tuple=(), _op: str=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other: 'int | float | Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(data=other)
        out = Value(data=self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other: 'int | float | Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other: 'int | float | Value') -> 'Value':
        assert isinstance(other, (int, float))
        out = Value(data=self.data**other, _children=(self,), _op=f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self) -> 'Value':
        return self * -1

    def __radd__(self, other: 'int | float | Value') -> 'Value':
        return self + other

    def __sub__(self, other: 'int | float | Value') -> 'Value':
        return self + (-other)

    def __rsub__(self, other: 'int | float | Value') -> 'Value':
        return other + (-self)

    def __rmul__(self, other: 'int | float | Value') -> 'Value':
        return self * other

    def __truediv__(self, other: 'int | float | Value') -> 'Value':
        return self * other**-1

    def __rtruediv__(self, other: 'int | float | Value') -> 'Value':
        return other * self**-1

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def exp(self) -> 'Value':
        out = Value(data=math.exp(self.data), _children=(self,), _op='Exp')

        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out._backward = _backward

        return out

    def relu(self) -> 'Value':
        out = Value(data=0 if self.data < 0 else self.data, _children=(self,), _op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self) -> 'Value':
        y = 2 * self
        t = (y.exp() - 1) / (y.exp() + 1)
        out = Value(data=t.data, _children=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t.data ** 2) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        topo = []
        visited = set()

        def build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()