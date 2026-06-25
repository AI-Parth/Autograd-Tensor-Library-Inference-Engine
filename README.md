# Autograd Tensor Library & Inference Engine

A from-scratch **automatic differentiation engine** written in modern C++20 — a miniature PyTorch core built without any external ML libraries. It implements dynamic computation graph construction, reverse-mode backpropagation, tensor operations, and gradient-based optimizers entirely using the C++ STL.

---

## Features

- **Dynamic computation graph** — the DAG is built on-the-fly during the forward pass via `creator` pointers on each `Tensor`
- **Reverse-mode autodiff** — `GraphEngine` performs DFS post-order topological sort and calls `backward()` in reverse order to propagate gradients via the chain rule
- **Tensor abstraction** — row-major contiguous `std::vector<float>` storage with shape, stride, and gradient buffer
- **Differentiable operations** — `MatMul`, `Add` (with broadcasting), `ReLU`, `MSELoss`
- **Optimizers** — SGD and Randomized Coordinate Descent
- **Singleton MemoryPool** — automatic lifetime management of heap-allocated intermediate tensors and ops; no smart-pointer overhead in the hot path
- **Zero external dependencies** — only the C++20 STL

---

## Architecture & OOP Design

The codebase is structured around five clean abstractions:

```
include/
  Tensor.h          Core data structure: data[], grad[], shape, creator*
  Operation.h       Abstract base class — forward() + backward() interface
  GraphEngine.h     DAG traversal + topological sort + backward pass driver
  MemoryPool.h      Singleton memory manager for intermediate allocations
  Optimizer.h       SGD and RandomizedCoordinateDescent
  autograd.h        Master header + factory function API (add, matmul, relu, mse_loss)
  ops/
    Add.h           Element-wise add with broadcast [m,n]+[n] support
    MatMul.h        Matrix×Matrix, Matrix×Vector, dot product
    ReLU.h          Element-wise max(0, x)
    MSELoss.h       Mean Squared Error → scalar loss
src/                .cpp implementations for all of the above
main.cpp            End-to-end demo: 1-layer linear+ReLU network, trained with SGD
Makefile            Single-command build
```

### Design Patterns

| Pattern | Where | Why |
|---|---|---|
| **Abstract Base Class / Polymorphism** | `Operation` → `AddOp`, `MatMulOp`, `ReLUOp`, `MSELossOp` | `GraphEngine` calls `backward()` on any op without knowing its type |
| **Composite** | `Tensor` (leaf vs. intermediate) | Graph traversal uses a uniform interface; `creator == nullptr` marks a leaf |
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

## Demo: Single Linear Layer with ReLU

`main.cpp` trains the model `Y_hat = ReLU(X·W + B)` for 20 SGD steps:

```
X  : [4 samples × 2 features]   — input data, no gradient
W  : [2 features × 3 outputs]   — learnable weight matrix
B  : [3 outputs]                 — learnable bias (broadcast)
Y_hat = ReLU( X·W + B )         — [4 × 3] predictions
Loss  = MSE(Y_hat, Y_target)     — scalar
```

**Training loop per step:**
```cpp
optimizer.zero_grad();                    // 1. clear W.grad, B.grad

Tensor* Z    = matmul(&X, &W);           // 2. forward pass
Tensor* ZB   = add(Z, &B);
Tensor* A    = relu(ZB);
Tensor* loss = mse_loss(A, &Y_target);

GraphEngine::backward(loss);             // 3. backprop
optimizer.step();                        // 4. W -= lr * W.grad, B -= lr * B.grad

MemoryPool::instance().cleanup();        // 5. free Z, ZB, A, loss + all Ops
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
    src/Optimizer.cpp src/ops/Add.cpp src/ops/MatMul.cpp \
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

## Optimizers

**SGD** — standard gradient descent:
```
param[i] -= learning_rate * grad[i]
```

**RandomizedCoordinateDescent** — updates only a random `fraction` of coordinates per step, which can be advantageous for high-dimensional, sparse gradient settings.

---

## Extending the Engine

To add a new differentiable operation:

1. Create `include/ops/MyOp.h` — subclass `Operation`, declare `forward()` and `backward()`
2. Create `src/ops/MyOp.cpp` — implement the math and gradient
3. Add a factory function in `autograd.h` following the existing pattern
4. Add the new `.cpp` to `SRCS` in `Makefile`
