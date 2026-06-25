#pragma once

#include <vector>

// Forward declarations — MemoryPool only stores pointers; full
// definitions are not needed in this header.
class Tensor;
class Operation;

/**
 * ============================================================
 *  MemoryPool  —  Lightweight memory manager (Singleton)
 * ============================================================
 *
 * PROBLEM:
 *   Every time we compute Y = matmul(X, W) + B, we heap-allocate
 *   intermediate Tensor and Operation objects.  Without central
 *   tracking we would leak them.
 *
 * SOLUTION:
 *   A Singleton pool that factory functions register objects with.
 *   At the end of each training iteration, call cleanup() to free
 *   every tracked object in one shot.
 *
 * DESIGN PATTERN: Singleton (Meyers' implementation)
 *   Only ONE MemoryPool instance exists for the entire program.
 *   Accessed via  MemoryPool::instance().
 *   C++11 guarantees that the local-static is initialised exactly
 *   once and is thread-safe.
 *
 * WHAT IS TRACKED:
 *   - Intermediate Tensor objects  (outputs of Ops)
 *   - Operation objects            (Add, MatMul, ReLU, MSELoss, …)
 *
 * WHAT IS NOT TRACKED:
 *   - Leaf Tensors created directly by the user (X, W, B, targets)
 *     Those are managed by the user (stack or their own heap).
 *
 * USAGE EXAMPLE:
 *   // Inside a factory function:
 *   Tensor*    out = new Tensor(...);
 *   AddOp*     op  = new AddOp(a, b, out);
 *   MemoryPool::instance().registerTensor(out);
 *   MemoryPool::instance().registerOp(op);
 *
 *   // End of training iteration:
 *   MemoryPool::instance().cleanup();   // frees out and op
 * ============================================================
 */
class MemoryPool {
public:
    /** Return the single global instance. */
    static MemoryPool& instance();

    /** Track a heap-allocated Tensor so cleanup() can free it. */
    void registerTensor(Tensor* t);

    /** Track a heap-allocated Operation so cleanup() can free it. */
    void registerOp(Operation* op);

    /**
     * Free every tracked Tensor and Operation, then clear the lists.
     *
     * Call this at the END of each training iteration, AFTER
     * optimizer.step() has applied the gradients.  This frees all
     * intermediate computation-graph nodes while leaving the user's
     * leaf Tensors (parameters, inputs) intact.
     */
    void cleanup();

    // ---- Singleton enforcement --------------------------------
    // Copying or assigning a Singleton breaks the "exactly one
    // instance" invariant, so we delete those special members.
    MemoryPool(const MemoryPool&)            = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

private:
    MemoryPool() = default;  // private → prevents external construction

    std::vector<Tensor*>    tensors;
    std::vector<Operation*> ops;
};