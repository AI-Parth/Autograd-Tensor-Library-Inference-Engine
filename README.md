# Autograd Tensor Library & Inference Engine

A **lightweight, fully educational autograd engine** built from scratch in pure **C++20** — no external dependencies beyond the standard library.

Inspired by micrograd and PyTorch internals, this project implements a complete computational graph with automatic differentiation, written to be read and understood by a second-year CS student.

---

## Features

| Component | Description |
|---|---|
| **Tensor** | Flat `std::vector<float>` storage, 1-D and 2-D shapes, row-major indexing |
| **AddOp** | Element-wise addition with `[m,n] + [n]` broadcasting support |
| **MatMulOp** | Matrix×Matrix, Matrix×Vector, and dot-product (Vector×Vector) |
| **ReLUOp** | Rectified Linear Unit with saved activation mask for efficient backward |
| **MSELossOp** | Mean Squared Error loss with analytical gradient |
| **GraphEngine** | DFS topological sort + reverse-order backward pass (chain rule) |
| **MemoryPool** | Singleton memory manager — tracks & frees all intermediate nodes |
| **SGD** | Stochastic Gradient Descent (`param -= lr * grad`) |
| **RandomizedCoordinateDescent** | Skeleton with documented TODO for future extension |

---

## Project Structure

```
.
├── include/
│   ├── autograd.h          # Master include + factory functions (add, matmul, relu, mse_loss)
│   ├── Tensor.h            # Core data structure
│   ├── Operation.h         # Abstract base class for all ops
│   ├── GraphEngine.h       # Topological sort + backward pass
│   ├── MemoryPool.h        # Singleton memory manager
│   ├── Optimizer.h         # SGD + RCD skeleton
│   └── ops/
│       ├── Add.h
│       ├── MatMul.h
│       ├── ReLU.h
│       └── MSELoss.h
├── src/
│   ├── Tensor.cpp
│   ├── GraphEngine.cpp
│   ├── MemoryPool.cpp
│   ├── Optimizer.cpp
│   └── ops/
│       ├── Add.cpp
│       ├── MatMul.cpp
│       ├── ReLU.cpp
│       └── MSELoss.cpp
├── main.cpp                # End-to-end training demo
└── Makefile
```

---

## Build & Run

```bash
make          # compiles with g++ -std=c++20 -Wall -Wextra -O2
make run      # compile + execute
make clean    # remove binary
```

Or manually:

```bash
g++ -std=c++20 -Wall -O2 -o autograd_engine \
    src/Tensor.cpp src/GraphEngine.cpp src/MemoryPool.cpp \
    src/Optimizer.cpp src/ops/Add.cpp src/ops/MatMul.cpp \
    src/ops/ReLU.cpp src/ops/MSELoss.cpp main.cpp
./autograd_engine
```

---

## Demo (main.cpp)

Trains a single linear layer with ReLU activation:

```
Y_hat = ReLU( X · W + B )
loss  = MSE( Y_hat, Y_target )
```

```
---- Training (20 steps) ----
  Step  1  |  Loss = 1.563334
  Step  5  |  Loss = 1.499588
  Step 10  |  Loss = 1.439658
  Step 20  |  Loss = 1.368255
```

---

## Design Patterns Used

| Pattern | Where |
|---|---|
| **Composite** | `Tensor` — leaf and intermediate nodes share the same interface |
| **Strategy** | `Operation` base class — each op encapsulates its own forward/backward |
| **Singleton** | `MemoryPool` — one global instance manages all heap allocations |

---

## Mathematical Background

### Chain Rule (the heart of backprop)

For a composed function `L = f(g(x))`:

```
dL/dx = (dL/df) * (df/dg) * (dg/dx)
```

The GraphEngine applies this by executing `backward()` on each Operation in **reverse topological order**: the downstream gradient (`output->grad`) is fully ready before an Op reads it.

### Matrix Multiplication Gradients

For `Z = A · B`:

```
dL/dA = dL/dZ · Bᵀ
dL/dB = Aᵀ   · dL/dZ
```

### Broadcast Gradient Reduction

When bias `B` of shape `[n]` is added to `A` of shape `[m, n]`, each `B[c]` contributes to `m` output elements. The backward pass sums those contributions:

```
dL/dB[c] = Σ_r  dL/dZ[r, c]
```

---

## Extending the Library

1. **New operation**: inherit from `Operation`, implement `forward()` and `backward()`, add a factory function to `autograd.h`.
2. **New optimizer**: add a class to `Optimizer.h` / `src/Optimizer.cpp` following the `SGD` pattern.
3. **Higher-rank tensors**: the flat-vector storage already supports N-D tensors; extend `strides()` and the Op shape logic as needed.
