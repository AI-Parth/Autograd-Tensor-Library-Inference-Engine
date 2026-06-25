#pragma once

#include "../Operation.h"

/**
 * ============================================================
 *  AddOp  —  Element-wise addition with optional broadcasting
 * ============================================================
 *
 * SUPPORTED SHAPES:
 *   [m, n] + [m, n]  →  plain element-wise add
 *   [m, n] + [n]     →  B is broadcast across the m rows
 *
 * ---- FORWARD -----------------------------------------------
 *   output[r][c] = A[r][c] + B[c]      (broadcast case)
 *   output[i]    = A[i]    + B[i]      (same-shape case)
 *
 * ---- BACKWARD (chain rule) ---------------------------------
 *   Let L be the scalar loss and Z = A + B.
 *
 *   dL/dA[r][c] = dL/dZ[r][c]         (gradient passes through addition)
 *
 *   dL/dB[c]    = Σ_r dL/dZ[r][c]     (broadcast case)
 *               = dL/dZ[c]            (same-shape case)
 *
 *   WHY sum over rows for the broadcast gradient?
 *   During forward, B[c] was ADDED to every row r.  Each of those m
 *   additions independently contributed to L.  The chain rule says
 *   we sum up all those contributions:
 *     dL/dB[c] = Σ_r (dL/d Z[r][c]) * (d Z[r][c] / d B[c])
 *              = Σ_r dL/dZ[r][c] * 1
 *              = Σ_r dL/dZ[r][c]
 * ============================================================
 */
class AddOp : public Operation {
public:
    /**
     * @param a      Left operand  (e.g. result of matmul, shape [m,n])
     * @param b      Right operand (e.g. bias,  shape [n] or [m,n])
     * @param output Pre-allocated output tensor (same shape as A)
     */
    AddOp(Tensor* a, Tensor* b, Tensor* output);

    void forward()  override;
    void backward() override;

private:
    bool broadcast; // true when B has fewer dims than A and must be tiled
};
