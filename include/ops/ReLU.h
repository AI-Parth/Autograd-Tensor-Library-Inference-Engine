#pragma once

#include "../Operation.h"
#include <vector>

/**
 * ============================================================
 *  ReLUOp  —  Rectified Linear Unit
 * ============================================================
 *
 * ---- FORWARD -----------------------------------------------
 *   output[i] = max(0, input[i])
 *
 *   Works identically for 1-D and 2-D tensors because both are
 *   stored as a flat vector — no dimension-specific code needed.
 *
 * ---- BACKWARD (subgradient) --------------------------------
 *   dL/dinput[i] = dL/doutput[i]   if input[i] > 0   (gate is open)
 *                = 0               if input[i] ≤ 0   (gate is shut)
 *
 *   This is implemented via a binary MASK saved during forward:
 *     mask[i] = 1.0f  if input[i] > 0
 *             = 0.0f  otherwise
 *
 *   Then backward is simply:
 *     in_grad[i] = out_grad[i] * mask[i]
 *
 *   Saving the mask avoids re-running forward during backward
 *   (which could have side-effects or be expensive in general).
 * ============================================================
 */
class ReLUOp : public Operation {
public:
    /**
     * @param input  Input tensor (any shape, stored flat)
     * @param output Pre-allocated output tensor (same shape)
     */
    ReLUOp(Tensor* input, Tensor* output);

    void forward()  override;
    void backward() override;

private:
    // Binary activation gate, computed during forward(), reused in backward().
    // mask[i] = 1.0f → gradient flows through; 0.0f → gradient is blocked.
    std::vector<float> mask;
};
