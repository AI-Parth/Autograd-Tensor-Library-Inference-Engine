#pragma once

#include <vector>
#include <string>
#include "Tensor.h"

/**
 * ============================================================
 *  Operation  (Abstract Base Class)
 * ============================================================
 * Every differentiable node in the computation graph inherits
 * from this class.
 *
 * DESIGN PATTERN: Strategy
 *   Each concrete subclass encapsulates:
 *     - HOW to compute the output  (forward)
 *     - HOW to propagate gradients (backward)
 *
 * LIFECYCLE:
 *   1. User calls a factory function  (e.g. `add(A, B)`)
 *   2. Factory creates the Op, calls forward(), links output->creator = op
 *   3. During backward(), GraphEngine calls backward() in reverse order
 *   4. MemoryPool frees the Op at the end of the iteration
 *
 * OWNERSHIP:
 *   Operations do NOT own their input/output Tensor objects.
 *   They merely hold raw pointers for access during forward/backward.
 * ============================================================
 */
class Operation {
public:
    std::vector<Tensor*> inputs;  // Input tensors consumed by this Op
    Tensor*              output;  // Single output tensor produced by this Op

    /** Compute the output from the inputs. */
    virtual void forward()  = 0;

    /**
     * Propagate gradients backwards.
     *
     * Precondition: output->grad has already been populated by the
     * operation that used our output as its input.
     *
     * Each implementation reads output->grad and calls
     * input->accumulate_grad(…) for each input that requires_grad.
     */
    virtual void backward() = 0;

    virtual ~Operation() = default;

    /** Human-readable name for debugging. */
    const std::string& name() const { return op_name; }

protected:
    std::string op_name = "Operation";
};
