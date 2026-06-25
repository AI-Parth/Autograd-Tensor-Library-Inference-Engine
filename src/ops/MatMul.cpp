#include "../../include/ops/MatMul.h"
#include <stdexcept>
#include <algorithm>

// ============================================================
//  Constructor
// ============================================================
MatMulOp::MatMulOp(Tensor* a, Tensor* b, Tensor* output) {
    op_name      = "MatMul";
    inputs       = {a, b};
    this->output = output;
}

// ============================================================
//  Forward: Z = A * B
// ============================================================
void MatMulOp::forward() {
    Tensor* A = inputs[0];
    Tensor* B = inputs[1];

    // Always start from zero (avoid stale values from a previous call)
    std::fill(output->data.begin(), output->data.end(), 0.0f);

    size_t ndA = A->ndim();
    size_t ndB = B->ndim();

    if (ndA == 2 && ndB == 2) {
        // ---- Case A: [m,n] × [n,p] → [m,p] ----
        // Z[i,j] = Σ_k  A[i,k] * B[k,j]
        size_t m = A->shape[0], n = A->shape[1], p = B->shape[1];
        for (size_t i = 0; i < m; ++i)
            for (size_t k = 0; k < n; ++k)          // k-loop in middle for cache reuse
                for (size_t j = 0; j < p; ++j)
                    output->data[i * p + j] += A->data[i * n + k] * B->data[k * p + j];

    } else if (ndA == 2 && ndB == 1) {
        // ---- Case B: [m,n] × [n] → [m] ----
        // z[i] = Σ_j  A[i,j] * b[j]
        size_t m = A->shape[0], n = A->shape[1];
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j)
                output->data[i] += A->data[i * n + j] * B->data[j];

    } else if (ndA == 1 && ndB == 1) {
        // ---- Case C: [n] · [n] → scalar ----
        // z = Σ_i  a[i] * b[i]
        size_t n = A->shape[0];
        float  dot = 0.0f;
        for (size_t i = 0; i < n; ++i)
            dot += A->data[i] * B->data[i];
        output->data[0] = dot;

    } else {
        throw std::invalid_argument(
            "MatMulOp::forward: unsupported shape combination.");
    }
}

// ============================================================
//  Backward
// ============================================================
// MATH SUMMARY (Case A):
//   Z = A B
//   dL/dA = dL/dZ · B^T    (shape: [m,n])
//   dL/dB = A^T · dL/dZ    (shape: [n,p])
//
// MATH SUMMARY (Case B):
//   z = A b
//   dL/dA[i,j] = dL/dz[i] * b[j]          (outer product)
//   dL/db[j]   = Σ_i dL/dz[i] * A[i,j]   (A^T * dL/dz)
//
// MATH SUMMARY (Case C):
//   z = a · b  (scalar)
//   dL/da[i] = dL/dz * b[i]
//   dL/db[i] = dL/dz * a[i]
void MatMulOp::backward() {
    Tensor* A = inputs[0];
    Tensor* B = inputs[1];
    const std::vector<float>& dZ = output->grad;

    size_t ndA = A->ndim();
    size_t ndB = B->ndim();

    if (ndA == 2 && ndB == 2) {
        size_t m = A->shape[0], n = A->shape[1], p = B->shape[1];

        // dL/dA = dL/dZ · B^T
        // dL/dA[i,k] = Σ_j  dZ[i,j] * B[k,j]
        if (A->requires_grad) {
            std::vector<float> dA(m * n, 0.0f);
            for (size_t i = 0; i < m; ++i)
                for (size_t k = 0; k < n; ++k)
                    for (size_t j = 0; j < p; ++j)
                        dA[i * n + k] += dZ[i * p + j] * B->data[k * p + j];
            A->accumulate_grad(dA);
        }

        // dL/dB = A^T · dL/dZ
        // dL/dB[k,j] = Σ_i  A[i,k] * dZ[i,j]
        if (B->requires_grad) {
            std::vector<float> dB(n * p, 0.0f);
            for (size_t i = 0; i < m; ++i)
                for (size_t k = 0; k < n; ++k)
                    for (size_t j = 0; j < p; ++j)
                        dB[k * p + j] += A->data[i * n + k] * dZ[i * p + j];
            B->accumulate_grad(dB);
        }

    } else if (ndA == 2 && ndB == 1) {
        size_t m = A->shape[0], n = A->shape[1];

        // dL/dA[i,j] = dZ[i] * b[j]
        if (A->requires_grad) {
            std::vector<float> dA(m * n, 0.0f);
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    dA[i * n + j] = dZ[i] * B->data[j];
            A->accumulate_grad(dA);
        }

        // dL/db[j] = Σ_i  dZ[i] * A[i,j]
        if (B->requires_grad) {
            std::vector<float> dB(n, 0.0f);
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    dB[j] += dZ[i] * A->data[i * n + j];
            B->accumulate_grad(dB);
        }

    } else if (ndA == 1 && ndB == 1) {
        float dz = dZ[0];   // scalar gradient
        size_t n = A->shape[0];

        // dL/da[i] = dz * b[i]
        if (A->requires_grad) {
            std::vector<float> dA(n);
            for (size_t i = 0; i < n; ++i) dA[i] = dz * B->data[i];
            A->accumulate_grad(dA);
        }

        // dL/db[i] = dz * a[i]
        if (B->requires_grad) {
            std::vector<float> dB(n);
            for (size_t i = 0; i < n; ++i) dB[i] = dz * A->data[i];
            B->accumulate_grad(dB);
        }
    }
}