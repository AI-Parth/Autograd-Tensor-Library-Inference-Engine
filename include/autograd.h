#pragma once

/**
 * ============================================================
 *  autograd.h  —  Master include + public factory API
 * ============================================================
 *
 * Include this single header to get the entire autograd library:
 *
 *   #include "include/autograd.h"
 *
 * ---- FACTORY FUNCTIONS -------------------------------------
 *   Each factory function:
 *     1. Computes the correct output shape from the inputs
 *     2. Allocates a new output Tensor on the heap
 *     3. Allocates the concrete Operation on the heap
 *     4. Calls op->forward()  to fill the output
 *     5. Sets output->creator = op  (links output into the graph)
 *     6. Registers both with MemoryPool for automatic cleanup
 *     7. Returns a raw pointer to the output Tensor
 *
 *   The caller receives a Tensor* that is valid until
 *   MemoryPool::instance().cleanup() is called.
 *
 * ---- TYPICAL TRAINING ITERATION ----------------------------
 *   // (W, B are user-managed leaf Tensors — NOT in MemoryPool)
 *
 *   optimizer.zero_grad();
 *
 *   Tensor* z    = matmul(&X, &W);        // [m,n] × [n,p]
 *   Tensor* zb   = add(z, &B);            // broadcast add
 *   Tensor* a    = relu(zb);              // activation
 *   Tensor* loss = mse_loss(a, &target);  // scalar loss
 *
 *   GraphEngine::backward(loss);          // backprop
 *   optimizer.step();                     // update W, B
 *
 *   MemoryPool::instance().cleanup();     // free z, zb, a, loss and all Ops
 * ============================================================
 */

#include "Tensor.h"
#include "Operation.h"
#include "ops/Add.h"
#include "ops/MatMul.h"
#include "ops/ReLU.h"
#include "ops/MSELoss.h"
#include "GraphEngine.h"
#include "MemoryPool.h"
#include "Optimizer.h"

#include <stdexcept>

// ============================================================
//  add(a, b)  →  a + b  (with broadcasting support)
// ============================================================
inline Tensor* add(Tensor* a, Tensor* b) {
    // Output has the same shape as `a`.
    // For the broadcast case ([m,n] + [n]), the AddOp constructor
    // validates the shapes and sets the broadcast flag internally.
    bool out_rg = a->requires_grad || b->requires_grad;
    Tensor* output = new Tensor(a->shape, 0.0f, out_rg);
    output->is_leaf = false;

    AddOp* op = new AddOp(a, b, output);
    op->forward();
    output->creator = op;

    MemoryPool::instance().registerTensor(output);
    MemoryPool::instance().registerOp(op);
    return output;
}

// ============================================================
//  matmul(a, b)  →  matrix/vector multiplication
// ============================================================
inline Tensor* matmul(Tensor* a, Tensor* b) {
    // Determine output shape based on the number of dimensions.
    std::vector<size_t> out_shape;
    size_t ndA = a->ndim(), ndB = b->ndim();

    if (ndA == 2 && ndB == 2) {
        // [m,n] × [n,p] → [m,p]
        if (a->shape[1] != b->shape[0])
            throw std::invalid_argument(
                "matmul: inner dimensions must match ([m,n] × [n,p]).");
        out_shape = {a->shape[0], b->shape[1]};

    } else if (ndA == 2 && ndB == 1) {
        // [m,n] × [n] → [m]
        if (a->shape[1] != b->shape[0])
            throw std::invalid_argument(
                "matmul: inner dimensions must match ([m,n] × [n]).");
        out_shape = {a->shape[0]};

    } else if (ndA == 1 && ndB == 1) {
        // [n] × [n] → scalar stored as shape {1}
        if (a->shape[0] != b->shape[0])
            throw std::invalid_argument(
                "matmul: vectors must be the same length for dot product.");
        out_shape = {1};

    } else {
        throw std::invalid_argument(
            "matmul: unsupported shape combination. "
            "Supported: [m,n]×[n,p], [m,n]×[n], [n]×[n].");
    }

    bool out_rg = a->requires_grad || b->requires_grad;
    Tensor* output = new Tensor(out_shape, 0.0f, out_rg);
    output->is_leaf = false;

    MatMulOp* op = new MatMulOp(a, b, output);
    op->forward();
    output->creator = op;

    MemoryPool::instance().registerTensor(output);
    MemoryPool::instance().registerOp(op);
    return output;
}

// ============================================================
//  relu(input)  →  max(0, input)
// ============================================================
inline Tensor* relu(Tensor* input) {
    Tensor* output = new Tensor(input->shape, 0.0f, input->requires_grad);
    output->is_leaf = false;

    ReLUOp* op = new ReLUOp(input, output);
    op->forward();
    output->creator = op;

    MemoryPool::instance().registerTensor(output);
    MemoryPool::instance().registerOp(op);
    return output;
}

// ============================================================
//  mse_loss(predictions, target)  →  scalar MSE loss tensor
// ============================================================
inline Tensor* mse_loss(Tensor* predictions, Tensor* target) {
    if (predictions->size() != target->size())
        throw std::invalid_argument(
            "mse_loss: predictions and target must have the same number of elements.");

    // Loss is always a scalar: shape {1}
    Tensor* output = new Tensor({1}, 0.0f, predictions->requires_grad);
    output->is_leaf = false;

    MSELossOp* op = new MSELossOp(predictions, target, output);
    op->forward();
    output->creator = op;

    MemoryPool::instance().registerTensor(output);
    MemoryPool::instance().registerOp(op);
    return output;
}