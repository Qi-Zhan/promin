import promin as pm


# ══════════════════════════════════════════════════════════════════════
#  Value — micrograd-style autograd engine
# ══════════════════════════════════════════════════════════════════════

_value_counter = 0


def _next_id():
    global _value_counter
    _value_counter += 1
    return _value_counter


class Value:
    """
    Stores a single scalar value and its gradient.
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label or f"v{_next_id()}"

    def __repr__(self):
        return f"Value({self.label}={self.data:.4f}, grad={self.grad:.4f})"

    # ── Forward operations ──────────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            local_grad = (other * self.data ** (other - 1)) * out.grad
            self.grad += local_grad

        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            local_grad = (out.data > 0) * out.grad
            self.grad += local_grad

        out._backward = _backward

        return out

    def tanh(self):
        import math

        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            local_grad = (1 - t**2) * out.grad
            self.grad += local_grad

        out._backward = _backward

        return out

    # ── Backward pass ───────────────────────────────────────────────

    def backward(self):
        # Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed gradient
        self.grad = 1.0

        # Reverse-mode autodiff
        for v in reversed(topo):
            v._backward()

    # ── Convenience operators ───────────────────────────────────────

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1


def neuron_example():
    """
    A single neuron: o = tanh(x1*w1 + x2*w2 + b)
    Then backward() to compute gradients.
    """
    global _value_counter
    _value_counter = 0

    # ── Inputs ──
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")

    # ── Weights ──
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")

    # ── Bias ──
    b = Value(6.8813735870195432, label="b")

    # ── Forward pass ──
    x1w1 = x1 * w1
    x1w1.label = "x1w1"
    x2w2 = x2 * w2
    x2w2.label = "x2w2"
    s = x1w1 + x2w2
    s.label = "x1w1+x2w2"
    n = s + b
    n.label = "n"
    o = n.tanh()
    o.label = "o"

    o.backward()

    return o, [x1, x2, w1, w2, b, x1w1, x2w2, s, n, o]


# ══════════════════════════════════════════════════════════════════════
#  Direct execution
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n{'─' * 50}")
    print("Computation graph (forward):")
    o, nodes = neuron_example()
    print(f"  o = tanh(x1*w1 + x2*w2 + b) = {o.data:.4f}")
    print(f"\nGradients (backward):")
    for v in nodes:
        print(f"  {v.label:10s}  data={v.data:8.4f}  grad={v.grad:8.4f}")
