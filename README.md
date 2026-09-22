# Formative 1 — A Neural Network Library from Scratch

This project implements a small neural network library in NumPy, built up
chapter by chapter: a base `Module` interface, layers, activations, losses,
an optimizer, and finally a training loop that ties everything together.

```
your_submission/
├── nn/
│   ├── __init__.py
│   ├── module.py
│   ├── layers/
│   │   ├── __init__.py
│   │   └── linear.py
│   ├── activations/
│   │   ├── __init__.py
│   │   ├── relu.py
│   │   ├── sigmoid.py
│   │   └── softmax.py
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── cross_entropy_loss.py
│   │   └── categorical_cross_entropy_loss.py
│   └── optim/
│       ├── __init__.py
│       └── sgd.py
├── tests/
├── main.py
└── README.md
```

---

## `nn/module.py` — *Ch. 0.5*

Defines the `Module` base class that every layer and activation subclasses.
It sets the shared interface — `forward(x)`, `backward(grad_output)`, and
`parameters()` — so that any two modules can be chained together and driven
by the same training loop, regardless of what they compute internally.
Losses (`CrossEntropyLoss`, `CategoricalCrossEntropyLoss`) deliberately do
**not** subclass `Module`, since they sit at the start of the backward pass
and have a different `forward` signature (they take `predictions` and
`targets`, not just `x`).

## `nn/layers/linear.py` — *Ch. 1, extended in Ch. 2*

The fully connected layer, `z = xW + b`.

- **`__init__`** creates `W` with Xavier/Glorot uniform initialization
  (`limit = sqrt(6 / (in_features + out_features))`) rather than zeros, so
  that units don't all start symmetric and computing identical gradients.
  `b` starts at zero, which is safe since bias has no symmetry problem.
- **`forward`** computes `x @ W + b` and caches `x`, since the backward
  pass needs it to compute `dW`.
- **`backward`** applies the chain rule for a matrix multiply:
  `dW = xᵀ @ grad_output`, `db = sum(grad_output, axis=0)`, and it returns
  `grad_output @ Wᵀ` as the gradient to pass to the previous layer.
- **`parameters`** returns `[(W, dW), (b, db)]` so the optimizer can update
  both in a uniform way.

## `nn/activations/relu.py` — *Ch. 1 (forward), Ch. 2 (forward), Ch. 3 (backward)*

Elementwise `max(0, x)`. `forward` caches a boolean mask of which entries
were positive; `backward` multiplies `grad_output` by that mask, since
ReLU's derivative is 1 where the input was positive and 0 elsewhere.

## `nn/activations/sigmoid.py` — *Ch. 2, 4*

Elementwise `1 / (1 + e^-x)`, squashing values into `(0, 1)` — used as the
output activation for binary classification.

- **`forward`** avoids overflow by branching on the sign of `x`: for
  `x ≥ 0` it computes `1 / (1 + exp(-x))`, and for `x < 0` it computes the
  algebraically equivalent `exp(x) / (1 + exp(x))`, so `exp()` is never
  applied to a large positive number. It caches its own output, `a`.
- **`backward`** uses the identity that sigmoid's derivative is
  `a * (1 - a)`, so no extra exponential is needed — just the cached `a`
  multiplied elementwise by `grad_output`.

## `nn/activations/softmax.py` — *Ch. 4, 7*

Row-wise softmax, converting a batch of logits into a probability
distribution per row (used for multi-class classification).

- **`forward`** subtracts each row's max before exponentiating (the
  "max-subtraction trick") — this doesn't change the result mathematically,
  since a constant shift cancels in the ratio of exponentials, but it keeps
  `exp()` from overflowing on large logits. It caches its output.
- **`backward`** uses the softmax Jacobian identity
  `dx[i] = out[i] * (grad_output[i] - Σⱼ grad_output[j] * out[j])`, applied
  row-wise, instead of building the full `(C, C)` Jacobian per example —
  much cheaper and simpler.

## `nn/losses/cross_entropy_loss.py` — *Ch. 5, 6*

Binary cross-entropy loss, for a single output probability per example
(pairs naturally with `Sigmoid`).

- **`forward`** clips predictions away from exactly 0 or 1 before taking
  logs, so `log(0)` (→ `-inf` → `nan`) never happens, then averages
  `-[t·log(a) + (1-t)·log(1-a)]` over the batch.
- **`backward`** returns `(a - t) / (a·(1-a))`, divided by the batch size
  `m` to stay consistent with the averaged loss from `forward`. This is the
  start of the gradient chain, so it takes no `grad_output` argument.

## `nn/losses/categorical_cross_entropy_loss.py` — *Ch. 6, 8*

Categorical cross-entropy loss for one-hot multi-class targets (pairs
naturally with `Softmax`).

- **`forward`** clips predictions away from 0, then computes
  `-mean(Σ_c t_c · log(a_c))` — since targets are one-hot, only the
  log-probability of the correct class contributes per example.
- **`backward`** returns `-t / a`, divided by batch size `m`. Note this is
  the gradient with respect to the softmax *probabilities*, not the fused
  `(a - t)` shortcut some textbooks use for Softmax+CCE combined — so
  `Softmax.backward()` still needs to be called afterward in the normal
  chain, rather than substituted out.

## `nn/optim/sgd.py` — *Ch. 9*

Vanilla stochastic gradient descent: `param -= lr * grad` for every tracked
parameter.

- Stores the `(param, grad)` pairs passed in **by reference**, not by
  copying, so that `step()` and `zero_grad()` mutate the same arrays the
  owning layer holds.
- **`step()`** subtracts `lr * grad` from each `param` in place.
- **`zero_grad()`** resets each `grad` array's contents to zero in place
  (`grad[...] = 0`), rather than rebinding the local name.
- **Gotcha to know about:** `Linear.backward()` *reassigns* `self.dW` and
  `self.db` to brand-new arrays on every call rather than updating them in
  place. That means an `SGD` built once, before training starts, will keep
  pointing at stale (or nonexistent) gradient arrays forever. The training
  loop in `main.py` works around this by refreshing
  `optimizer.parameters = layer.parameters()` after every `backward()`
  call, right before `step()`.

## `main.py` — *Ch. 10*

The end-to-end training script that wires everything above together:
builds a small network (e.g. `Linear → Sigmoid`, or `Linear → ReLU →
Linear → Sigmoid` for problems that aren't linearly separable), trains it
with `SGD` on a toy dataset, and reports loss and accuracy. Structured so
that `tests/test_stage10_training_converges.py` can `import main` and call
`main.toy_data()`, `main.train(...)`, and `main.accuracy()` directly. The
chapter note ("toy data, then real data") suggests this file is meant to
later be extended to train on a real dataset once the toy XOR/AND-style
example is passing.

## `tests/` — *provided, do not edit*

The test suite used to check each chapter's implementation. These files
are provided as-is and shouldn't be modified — they're the spec your `nn/`
modules and `main.py` need to satisfy.

## `README.md`

This file.
