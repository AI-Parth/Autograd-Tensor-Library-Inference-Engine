#include "../../include/ops/MSELoss.h"
#include <stdexcept>

// ============================================================
//  Constructor
// ============================================================
MSELossOp::MSELossOp(Tensor* predictions, Tensor* target, Tensor* output) {
    if (predictions->size() != target->size())
        throw std::invalid_argument(
            "MSELossOp: predictions and target must contain the same number of elements.");
    op_name      = "MSELoss";
    inputs       = {predictions, target};
    this->output = output;
}

// ============================================================
//  Forward: loss = (1/N) * Σ_i (pred[i] - target[i])^2
// ============================================================
void MSELossOp::forward() {
    Tensor* pred   = inputs[0];
    Tensor* target = inputs[1];
    size_t  N      = pred->size();

    float sum = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float e = pred->data[i] - target->data[i];  // prediction error
        sum    += e * e;
    }
    output->data[0] = sum / static_cast<float>(N);
}

// ============================================================
//  Backward: dL/dpred[i] = (2/N) * (pred[i] - target[i])
// ============================================================
// DERIVATION:
//   loss = (1/N) Σ_i e_i^2   where e_i = pred_i - target_i
//
//   d(loss)/d(pred_i) = (1/N) * 2 * e_i * (d e_i / d pred_i)
//                     = (2/N) * e_i * 1
//                     = (2/N) * (pred_i - target_i)
//
// With the incoming gradient dLoss = output->grad[0] (= 1.0 for the
// root loss node), the full chain-rule gradient is:
//
//   grad_pred[i] = dLoss * (2/N) * (pred_i - target_i)
//
// The target tensor is treated as a constant — no gradient needed.
void MSELossOp::backward() {
    Tensor* pred   = inputs[0];
    Tensor* target = inputs[1];
    if (!pred->requires_grad) return;

    size_t N     = pred->size();
    float  dLoss = output->grad[0];   // incoming scalar gradient

    std::vector<float> grad_pred(N);
    for (size_t i = 0; i < N; ++i)
        grad_pred[i] = dLoss * (2.0f / static_cast<float>(N))
                              * (pred->data[i] - target->data[i]);

    pred->accumulate_grad(grad_pred);
}
