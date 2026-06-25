#include "../../include/ops/Add.h"
#include <stdexcept>

// ============================================================
//  Constructor
// ============================================================
AddOp::AddOp(Tensor* a, Tensor* b, Tensor* output)
    : broadcast(false)
{
    op_name = "Add";
    inputs  = {a, b};
    this->output = output;

    // Decide at construction time whether broadcasting is needed.
    // We check shapes once here so forward() and backward() don't
    // repeat the shape logic.
    if (a->shape != b->shape) {
        // We support exactly: [m, n] + [n]
        if (a->ndim() == 2 && b->ndim() == 1 && a->shape[1] == b->shape[0]) {
            broadcast = true;
        } else {
            throw std::invalid_argument(
                "AddOp: incompatible shapes. "
                "Supported: same-shape OR [m,n]+[n] broadcasting.");
        }
    }
}

// ============================================================
//  Forward: output = A + B
// ============================================================
void AddOp::forward() {
    Tensor* a = inputs[0];
    Tensor* b = inputs[1];

    if (!broadcast) {
        // Plain element-wise addition (shapes are identical)
        for (size_t i = 0; i < output->size(); ++i)
            output->data[i] = a->data[i] + b->data[i];

    } else {
        // Broadcast: A is [m, n], B is [n].
        // Each row of A gets B added to it element-wise.
        // output[r][c] = A[r][c] + B[c]
        size_t rows = a->shape[0];
        size_t cols = a->shape[1];
        for (size_t r = 0; r < rows; ++r)
            for (size_t c = 0; c < cols; ++c)
                output->data[r * cols + c] = a->data[r * cols + c] + b->data[c];
    }
}

// ============================================================
//  Backward: accumulate gradients into A and B
// ============================================================
// dL/dA = dL/d(output)                      (gradient passes through)
// dL/dB = dL/d(output)          same-shape
//       = Σ_r dL/d(output)[r]   broadcast (reduce over rows)
void AddOp::backward() {
    Tensor* a = inputs[0];
    Tensor* b = inputs[1];
    const std::vector<float>& dZ = output->grad;

    // ---- Gradient for A -----
    // Addition is linear: dZ/dA = 1  element-wise.
    // So dL/dA = dL/dZ * 1 = dL/dZ.
    if (a->requires_grad)
        a->accumulate_grad(dZ);

    // ---- Gradient for B -----
    if (b->requires_grad) {
        if (!broadcast) {
            // Same shapes: gradient passes through directly.
            b->accumulate_grad(dZ);
        } else {
            // B was broadcast across rows.
            // Each B[c] contributed to output[0][c], output[1][c], …
            // so its total gradient is the SUM of those contributions.
            //   dL/dB[c] = Σ_r dL/dZ[r * cols + c]
            size_t rows   = a->shape[0];
            size_t cols   = a->shape[1];
            std::vector<float> b_grad(cols, 0.0f);
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < cols; ++c)
                    b_grad[c] += dZ[r * cols + c];
            b->accumulate_grad(b_grad);
        }
    }
}
