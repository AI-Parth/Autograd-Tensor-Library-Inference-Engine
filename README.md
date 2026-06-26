# Autograd Tensor Library & Inference Engine

A from-scratch **automatic differentiation engine** written in modern C++20 — a miniature deep-learning framework built without any external ML libraries. It implements dynamic computation graph construction, reverse-mode backpropagation, tensor operations, gradient-based optimizers, and a **PyTorch-style `flash::` module API** entirely using the C++ STL.

---

## Features

- **Dynamic computation graph** — the DAG is built on-the-fly during the forward pass via `creator` pointers on each `Tensor`
- **Reverse-mode autodiff** — `GraphEngine` performs DFS post-order topological sort and calls `backward()` in reverse order to propagate gradients via the chain rule
- **Tensor abstraction** — row-major contiguous `std::vector<float>` storage with shape, stride, and gradient buffer
- **Differentiable operations** — `MatMul`, `Add` (with broadcasting), `ReLU`, `MSELoss`
- **Optimizer hierarchy** — abstract `Optimizer` base class with `SGD`, `RandomizedCoordinateDescent`, and `Adam` subclasses
- **`nn` module system** — `Module`, `Linear`, `ReLULayer`, `Sequential` (mirrors `torch.nn`)
- **`flash::` API** — single-header `include/flash.h` with a `flash::` namespace for PyTorch-style usage
- **Singleton MemoryPool** — automatic lifetime management of heap-allocated intermediate tensors and ops; no smart-pointer overhead in the hot path
- **Zero external dependencies** — only the C++20 STL

---

## Quick Start — PyTorch-style `flash::` API

```cpp
#include "include/flash.h"

// 1. Build a model exactly like torch.nn.Sequential
flash::Sequential model;
model.add(std::make_unique<flash::Linear>(2, 8));
model.add(std::make_unique<flash::ReLULayer>());
model.add(std::make_unique<flash::Linear>(8, 3));

// 2. Create an Adam optimizer over all model parameters
flash::Adam optimizer(model.parameters(), /*lr=*/0.001f);

// 3. Training loop
for (int epoch = 0; epoch < 100; ++epoch) {
    optimizer.zero_grad();

    flash::Tensor* pred = model.forward(&X);
    flash::Tensor* loss = flash::mse_loss(pred, &Y_target);

    flash::GraphEngine::backward(loss);
    optimizer.step();
    flash::MemoryPool::instance().cleanup();
}
```

---

## Architecture & OOP Design

```
include/
  Tensor.h              Core data structure: data[], grad[], shape, creator*
  Operation.h           Abstract base class — forward() + backward() interface
  GraphEngine.h         DAG traversal + topological sort + backward pass driver
  MemoryPool.h          Singleton memory manager for intermediate allocations
  Optimizer.h           Optimizer ABC → SGD, RandomizedCoordinateDescent, Adam
  autograd.h            Low-level factory function API (add, matmul, relu, mse_loss)
  flash.h               ⚡ Single master header — exposes everything under flash::
  ops/
    Add.h               Element-wise add with broadcast [m,n]+[n] support
    MatMul.h            Matrix×Matrix, Matrix×Vector, dot product
    ReLU.h              Element-wise max(0, x)
    MSELoss.h           Mean Squared Error → scalar loss
  nn/
    Module.h            Abstract base class — forward() + parameters() interface
    Linear.h            Y = X·W + B  (randomly initialised weights)
    ReLULayer.h         Stateless ReLU activation layer
    Sequential.h        Ordered container of Module subclasses
src/                    .cpp implementations for all of the above
main.cpp                End-to-end demo: 2-layer network trained with Adam
Makefile                Single-command build
```

### Design Patterns

| Pattern | Where | Why |
|---|---|---|
| **Abstract Base Class / Polymorphism** | `Operation` → `AddOp`, `MatMulOp`, `ReLUOp`, `MSELossOp` | `GraphEngine` calls `backward()` on any op without knowing its type |
| **Abstract Base Class / Polymorphism** | `Optimizer` → `SGD`, `RandomizedCoordinateDescent`, `Adam` | Swap optimizers without changing training loop code |
| **Abstract Base Class / Polymorphism** | `Module` → `Linear`, `ReLULayer`, `Sequential` | Uniform `forward()` / `parameters()` interface across all layers |
| **Composite** | `Tensor` (leaf vs. intermediate) | Graph traversal uses a uniform interface; `creator == nullptr` marks a leaf |
| **Composite** | `Sequential` contains `vector<unique_ptr<Module>>` | Nests arbitrary layers; `forward()` and `parameters()` recurse automatically |
| **Strategy** | Each `Op` subclass owns its forward math + gradient math | Adding a new op requires no changes to the engine |
| **Singleton** | `MemoryPool` | One global pool registers and frees all intermediate allocations in one `cleanup()` call |
| **Factory Function** | `add()`, `matmul()`, `relu()`, `mse_loss()` in `autograd.h` | Hides object wiring (alloc → forward → link → register) behind a clean functional API |

---

## Computation Graph & Backpropagation

Every factory function:
1. Allocates an output `Tensor` and a concrete `Operation` on the heap
2. Calls `op->forward()` to compute the output values
3. Sets `output->creator = op` — linking the output into the graph
4. Registers both with `MemoryPool` for later cleanup

`GraphEngine::backward(loss)` then:
1. Seeds `loss->grad = {1.0f}` (i.e., dL/dL = 1)
2. Walks the graph via DFS post-order to collect all `Operation*` nodes in topological order
3. Iterates in **reverse** topological order, calling `op->backward()` on each — ensuring every downstream gradient is fully accumulated before an upstream op runs

Gradients are **accumulated** (not overwritten) via `accumulate_grad()`, correctly handling diamond-shaped graphs where one tensor feeds multiple operations.

---

## Optimizers

All optimizers inherit from the abstract `Optimizer` base class and share a common interface:

```cpp
optimizer.zero_grad();   // clear all parameter gradients
optimizer.step();        // apply one update step
```

### SGD — Stochastic Gradient Descent
```
param[i] -= learning_rate * grad[i]
```

### RandomizedCoordinateDescent
Updates only a random `fraction` of coordinates per step — advantageous for high-dimensional, sparse gradient settings.

### Adam — Adaptive Moment Estimation
```
m[i] = β₁·m[i] + (1−β₁)·grad[i]          // 1st moment (mean)
v[i] = β₂·v[i] + (1−β₂)·grad[i]²         // 2nd moment (variance)
m̂   = m[i] / (1−β₁ᵗ)                      // bias correction
v̂   = v[i] / (1−β₂ᵗ)
param[i] -= lr · m̂ / (√v̂ + ε)
```
Defaults: `β₁=0.9`, `β₂=0.999`, `ε=1e-8`, `lr=0.001`.

---

## `nn` Module System

### `Module` (abstract base)
```cpp
class Module {
public:
    virtual Tensor* forward(Tensor* input) = 0;
    virtual std::vector<Tensor*> parameters() = 0;
    void zero_grad();
};
```

### `Linear`
`Y = X·W + B` — weights randomly initialised in `[-0.1, 0.1]`, bias zero-initialised.
```cpp
flash::Linear layer(in_features, out_features);
```

### `ReLULayer`
Stateless activation layer wrapping the `relu()` factory function.

### `Sequential`
```cpp
flash::Sequential model;
model.add(std::make_unique<flash::Linear>(2, 8));
model.add(std::make_unique<flash::ReLULayer>());
model.add(std::make_unique<flash::Linear>(8, 3));

flash::Tensor* output = model.forward(&input);
std::vector<flash::Tensor*> params = model.parameters();
```

---

## Demo: 2-Layer Network with `flash::` API

`main.cpp` trains a 2-layer MLP `Y_hat = Linear(8,3)(ReLU(Linear(2,8)(X)))` for 100 Adam steps:

```
X        : [4 samples × 2 features]   — input data, no gradient
Y_target : [4 samples × 3 outputs]    — ground-truth labels
Model    : Linear(2→8) → ReLU → Linear(8→3)
Loss     : MSE(Y_hat, Y_target)        — scalar
```

**Training loop:**
```cpp
flash::Adam optimizer(model.parameters(), 0.001f);

for (int step = 0; step < 100; ++step) {
    optimizer.zero_grad();

    flash::Tensor* pred = model.forward(&X);
    flash::Tensor* loss = flash::mse_loss(pred, &Y_target);

    flash::GraphEngine::backward(loss);
    optimizer.step();
    flash::MemoryPool::instance().cleanup();
}
```

---

## Build & Run

**Requirements:** `g++` with C++20 support (GCC 10+ or Clang 13+), `make`

```bash
# Clone
git clone https://github.com/AI-Parth/Autograd-Tensor-Library-Inference-Engine.git
cd Autograd-Tensor-Library-Inference-Engine

# Build
make

# Run the demo
make run

# Clean
make clean
```

Or compile manually:
```bash
g++ -std=c++20 -Wall -Wextra -O2 -o autograd_engine \
    src/Tensor.cpp src/GraphEngine.cpp src/MemoryPool.cpp \
    src/Optimizer.cpp \
    src/nn/Linear.cpp src/nn/ReLULayer.cpp src/nn/Sequential.cpp \
    src/ops/Add.cpp src/ops/MatMul.cpp \
    src/ops/ReLU.cpp src/ops/MSELoss.cpp main.cpp
```

---

## Implemented Operations

| Operation | Forward | Backward (gradient w.r.t. inputs) |
|---|---|---|
| `MatMul` [m,n]×[n,p] | Z = A·B | dA = dZ·Bᵀ, dB = Aᵀ·dZ |
| `MatMul` [m,n]×[n] | z = A·b | dA = dz ⊗ b, db = Aᵀ·dz |
| `Add` (same shape) | Z = A + B | dA = dZ, dB = dZ |
| `Add` (broadcast [m,n]+[n]) | Z[r][c] = A[r][c] + B[c] | dA = dZ, dB[c] = Σᵣ dZ[r][c] |
| `ReLU` | Z = max(0, X) | dX[i] = dZ[i] if X[i] > 0, else 0 |
| `MSELoss` | L = (1/N)·Σ(ŷ−y)² | dŷ[i] = (2/N)·(ŷ[i]−y[i]) |

---

## Extending the Engine

### Adding a new differentiable operation
1. Create `include/ops/MyOp.h` — subclass `Operation`, declare `forward()` and `backward()`
2. Create `src/ops/MyOp.cpp` — implement the math and gradient
3. Add a factory function in `autograd.h` following the existing pattern
4. Add a `using ::my_op` alias in `flash.h`
5. Add the new `.cpp` to `SRCS` in `Makefile`

### Adding a new layer
1. Create `include/nn/MyLayer.h` — subclass `Module`, implement `forward()` and `parameters()`
2. Create `src/nn/MyLayer.cpp`
3. Add `#include "nn/MyLayer.h"` to `flash.h` and a `using ::MyLayer` alias
4. Add `src/nn/MyLayer.cpp` to `SRCS` in `Makefile`

### Adding a new optimizer
1. Subclass `Optimizer` in `include/Optimizer.h`
2. Implement `step()` in `src/Optimizer.cpp`
3. Add a `using ::MyOptimizer` alias in `flash.h`
