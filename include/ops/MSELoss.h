#pragma once

#include "../Operation.h"

/**
 * ============================================================
 *  MSELossOp  —  Mean Squared Error Loss
 * ============================================================
 *
 * ---- FORWARD -----------------------------------------------
 *   loss = (1/N) * Σ_i (pred[i] - target[i])^2
 *
 *   N      = number of elements in predictions
 *   output = scalar tensor of shape {1}
 *
 * ---- BACKWARD ----------------------------------------------
 *   We need dL/dpred[i] (gradient of loss w.r.t. each prediction).
 *
 *   Let e[i] = pred[i] - target[i]  (prediction error)
 *   loss = (1/N) Σ e[i]^2
 *
 *   d(loss)/d(pred[i]) = (2/N) * e[i]
 *                      = (2/N) * (pred[i] - target[i])
 *
 *   With the incoming gradient dLoss = output->grad[0] (= 1.0 for
 *   the final loss node), the chain rule gives:
 *
 *   grad_pred[i] = dLoss * (2/N) * (pred[i] - target[i])
 *
 *   The target tensor is a CONSTANT — no gradient flows to it.
 * ============================================================
 */
class MSELossOp : public Operation {
public:
    /**
     * @param predictions  Model output tensor (any shape, stored flat)
     * @param target       Ground-truth tensor (same shape, no grad needed)
     * @param output       Pre-allocated scalar output tensor, shape {1}
     */
    MSELossOp(Tensor* predictions, Tensor* target, Tensor* output);

    void forward()  override;
    void backward() override;
};
