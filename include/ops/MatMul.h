#pragma once

#include "../Operation.h"

/**
 * ============================================================
 *  MatMulOp  —  Matrix / Vector multiplication
 * ============================================================
 *
 * SUPPORTED CASES (determined at runtime from tensor shapes):
 *
 *   Case A  [m,n] × [n,p]  →  [m,p]   Matrix × Matrix
 *   Case B  [m,n] × [n]    →  [m]     Matrix × Vector
 *   Case C  [n]   × [n]    →  [1]     Vector dot product
 *
 * ---- FORWARD -----------------------------------------------
 *
 *   Case A:  Z[i,j] = Σ_k  A[i,k] * B[k,j]
 *   Case B:  z[i]   = Σ_j  A[i,j] * b[j]
 *   Case C:  z      = Σ_i  a[i]   * b[i]
 *
 * ---- BACKWARD (analytical gradients) ----------------------
 *
 *   Case A:  Z = A B
 *     dL/dA = dL/dZ  · B^T      ← (m×p) · (p×n) = (m×n)
 *     dL/dB = A^T    · dL/dZ    ← (n×m) · (m×p) = (n×p)
 *
 *     Derivation of dL/dA:
 *       L is a scalar, Z[i,j] = Σ_k A[i,k] B[k,j]
 *       dL/dA[i,k] = Σ_j (dL/dZ[i,j]) * (dZ[i,j]/dA[i,k])
 *                  = Σ_j (dL/dZ[i,j]) * B[k,j]
 *                  = (dL/dZ · B^T)[i,k]   ✓
 *
 *   Case B:  z = A b
 *     dL/dA[i,j] = dL/dz[i] * b[j]          (outer product)
 *     dL/db[j]   = Σ_i dL/dz[i] * A[i,j]   (A^T · dL/dz)
 *
 *   Case C:  z = a · b  (scalar)
 *     dL/da[i] = dL/dz * b[i]
 *     dL/db[i] = dL/dz * a[i]
 * ============================================================
 */
class MatMulOp : public Operation {
public:
    /**
     * @param a      Left operand
     * @param b      Right operand
     * @param output Pre-allocated output tensor (correct shape for case A/B/C)
     */
    MatMulOp(Tensor* a, Tensor* b, Tensor* output);

    void forward()  override;
    void backward() override;
};
