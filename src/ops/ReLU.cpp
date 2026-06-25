#include "../../include/ops/ReLU.h"

// ============================================================
//  Constructor
// ============================================================
ReLUOp::ReLUOp(Tensor* input, Tensor* output) {
    op_name      = "ReLU";
    inputs       = {input};
    this->output = output;
    // Pre-allocate the mask with the same number of elements as the input.
    // We fill it during forward() and reuse it in backward().
    mask.resize(input->size(), 0.0f);
}

// ============================================================
//  Forward: output[i] = max(0, input[i])
// ============================================================
// Simultaneously record the binary mask for the backward pass.
// mask[i] = 1.0f  →  the gate is open   (positive activation)
// mask[i] = 0.0f  →  the gate is shut   (negative activation)
//
// Works identically for 1-D and 2-D tensors because both are
// stored as a flat vector — no dimension-specific branching needed.
void ReLUOp::forward() {
    Tensor* input = inputs[0];
    for (size_t i = 0; i < input->size(); ++i) {
        if (input->data[i] > 0.0f) {
            output->data[i] = input->data[i];
            mask[i]         = 1.0f;
        } else {
            output->data[i] = 0.0f;
            mask[i]         = 0.0f;
        }
    }
}

// ============================================================
//  Backward: dL/dinput[i] = dL/doutput[i] * mask[i]
// ============================================================
// The mask gates the gradient:
//   If the neuron was active (input > 0), gradient flows through unchanged.
//   If the neuron was inactive (input ≤ 0), gradient is blocked (= 0).
void ReLUOp::backward() {
    Tensor* input = inputs[0];
    if (!input->requires_grad) return;

    const std::vector<float>& out_grad = output->grad;
    std::vector<float> in_grad(input->size());
    for (size_t i = 0; i < input->size(); ++i)
        in_grad[i] = out_grad[i] * mask[i];

    input->accumulate_grad(in_grad);
}