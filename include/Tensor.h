#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <numeric>

// Forward declaration — breaks the circular dependency between Tensor and Operation.
// Tensor needs to *point* to an Operation, but does not need its full definition here.
class Operation;

/**
 * ============================================================
 *  Tensor
 * ============================================================
 * The fundamental data structure of the autograd engine.
 *
 * DESIGN PATTERN: Composite
 *   A Tensor looks the same whether it is a LEAF node (created
 *   directly by the user) or an INTERMEDIATE node (produced by
 *   an Operation).  The only difference is whether `creator` is
 *   nullptr or points to the Op that produced it.
 *
 * MEMORY LAYOUT:
 *   All values live in a single contiguous std::vector<float>.
 *   For a 2D tensor of shape {rows, cols}, element (r, c) lives
 *   at flat index:  r * cols + c   (row-major / C-style layout).
 *
 *   This is cache-friendly and avoids pointer-chasing.
 *
 * EXAMPLE:
 *   Tensor A({2, 3}, 0.0f);   // 2×3 matrix, zero-filled
 *   A.at(1, 2) = 5.0f;        // set element at row=1, col=2
 *   size_t n = A.size();      // → 6
 * ============================================================
 */
class Tensor {
public:
    // ---- Core storage ----------------------------------------
    std::vector<float>  data;   // Flat array of all values
    std::vector<float>  grad;   // Gradient buffer (same length as data)
    std::vector<size_t> shape;  // Logical shape, e.g. {batch, features}

    // ---- Computation graph metadata --------------------------
    Operation* creator;    // Op that produced this tensor; nullptr = leaf
    bool requires_grad;    // True → participate in gradient computation
    bool is_leaf;          // True → created directly by the user

    // ---- Constructors ----------------------------------------

    /** Default: empty tensor (no data, no shape). */
    Tensor();

    /**
     * Construct from an existing flat data vector.
     *
     * @param data          Values; must equal the product of shape dims.
     * @param shape         e.g. {3} for a length-3 vector, {2,4} for 2×4 matrix.
     * @param requires_grad Whether gradients should be computed for this tensor.
     */
    Tensor(std::vector<float>  data,
           std::vector<size_t> shape,
           bool                requires_grad = false);

    /**
     * Construct a tensor filled with a constant value.
     *
     * @param shape         Logical shape.
     * @param fill_value    Every element is initialised to this (default 0).
     * @param requires_grad Whether gradients should be computed for this tensor.
     */
    explicit Tensor(std::vector<size_t> shape,
                    float               fill_value    = 0.0f,
                    bool                requires_grad = false);

    // ---- Shape helpers ---------------------------------------

    /** Total number of elements = product of all dimensions. */
    size_t size() const;

    /** Number of dimensions (1 for vector, 2 for matrix, …). */
    size_t ndim() const;

    /**
     * Row-major strides for arbitrary-rank indexing.
     * stride[i] = product of shape[i+1 … ndim-1].
     * For shape {2, 3}:  strides = {3, 1}.
     * For shape {2,3,4}: strides = {12, 4, 1}.
     */
    std::vector<size_t> strides() const;

    // ---- Element access --------------------------------------

    /** 1-D access: data[i]. */
    float&       at(size_t i);
    const float& at(size_t i) const;

    /** 2-D access: data[row * cols + col]. */
    float&       at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;

    // ---- Gradient helpers ------------------------------------

    /** Set every gradient element to zero. Call before each backward pass. */
    void zero_grad();

    /**
     * ADD `incoming_grad` into this tensor's gradient buffer.
     *
     * WHY ADD instead of overwrite?
     *   In a computation graph, multiple paths can reach the same tensor.
     *   Each path contributes its own partial gradient.  The chain rule
     *   says the total gradient is the SUM of all path contributions.
     *
     * Example: if tensor T is used in two operations Op1 and Op2,
     *   T.grad = grad_from_Op1 + grad_from_Op2.
     */
    void accumulate_grad(const std::vector<float>& incoming_grad);

    // ---- Debug / display -------------------------------------
    void print(const std::string& name = "Tensor") const;
    void print_grad(const std::string& name = "Tensor") const;

    // ---- Static helpers --------------------------------------
    /** Throws if a.shape != b.shape. */
    static void validate_shapes_equal(const Tensor& a, const Tensor& b);
};
