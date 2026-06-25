#include "../include/Tensor.h"
#include <stdexcept>
#include <iomanip>
#include <numeric>

// ============================================================
//  Default constructor
// ============================================================
Tensor::Tensor()
    : creator(nullptr), requires_grad(false), is_leaf(true) {}

// ============================================================
//  Construct from existing flat data
// ============================================================
Tensor::Tensor(std::vector<float>  data,
               std::vector<size_t> shape,
               bool                requires_grad)
    : data(std::move(data))
    , grad(this->data.size(), 0.0f)
    , shape(std::move(shape))
    , creator(nullptr)
    , requires_grad(requires_grad)
    , is_leaf(true)
{
    // Validate: the number of data elements must match shape dimensions.
    size_t expected = 1;
    for (size_t d : this->shape) expected *= d;
    if (expected != this->data.size()) {
        throw std::invalid_argument(
            "Tensor: data.size() = " + std::to_string(this->data.size()) +
            " does not match shape product = " + std::to_string(expected) + ".");
    }
}

// ============================================================
//  Construct zero/fill tensor
// ============================================================
Tensor::Tensor(std::vector<size_t> shape,
               float               fill_value,
               bool                requires_grad)
    : shape(std::move(shape))
    , creator(nullptr)
    , requires_grad(requires_grad)
    , is_leaf(true)
{
    size_t n = 1;
    for (size_t d : this->shape) n *= d;
    data.assign(n, fill_value);
    grad.assign(n, 0.0f);
}

// ============================================================
//  Shape helpers
// ============================================================

size_t Tensor::size() const {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

size_t Tensor::ndim() const {
    return shape.size();
}

// Row-major strides:
//   stride[i] = product of shape[i+1 … ndim-1]
// For shape {2, 3}:      strides = {3, 1}
// For shape {2, 3, 4}:   strides = {12, 4, 1}
std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> s(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        s[static_cast<size_t>(i)] = s[static_cast<size_t>(i) + 1] * shape[static_cast<size_t>(i) + 1];
    }
    return s;
}

// ============================================================
//  Element access
// ============================================================

float& Tensor::at(size_t i) {
    return data.at(i);
}

const float& Tensor::at(size_t i) const {
    return data.at(i);
}

float& Tensor::at(size_t row, size_t col) {
    if (shape.size() != 2)
        throw std::runtime_error("Tensor::at(row, col): tensor must be 2-D.");
    return data.at(row * shape[1] + col);
}

const float& Tensor::at(size_t row, size_t col) const {
    if (shape.size() != 2)
        throw std::runtime_error("Tensor::at(row, col): tensor must be 2-D.");
    return data.at(row * shape[1] + col);
}

// ============================================================
//  Gradient helpers
// ============================================================

void Tensor::zero_grad() {
    std::fill(grad.begin(), grad.end(), 0.0f);
}

// Accumulate (ADD) incoming_grad into this tensor's grad buffer.
// We add rather than assign because multiple graph paths can each
// contribute a partial gradient to the same tensor.
void Tensor::accumulate_grad(const std::vector<float>& incoming_grad) {
    if (incoming_grad.size() != grad.size()) {
        throw std::runtime_error(
            "Tensor::accumulate_grad: size mismatch. "
            "incoming=" + std::to_string(incoming_grad.size()) +
            " vs grad=" + std::to_string(grad.size()) + ".");
    }
    for (size_t i = 0; i < grad.size(); ++i)
        grad[i] += incoming_grad[i];
}

// ============================================================
//  Printing
// ============================================================

static void print_flat(const std::vector<float>& v,
                       const std::vector<size_t>& shape) {
    if (shape.size() == 1) {
        std::cout << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << v[i];
            if (i + 1 < v.size()) std::cout << ", ";
        }
        std::cout << "]";
    } else if (shape.size() == 2) {
        size_t rows = shape[0], cols = shape[1];
        for (size_t r = 0; r < rows; ++r) {
            std::cout << "  [";
            for (size_t c = 0; c < cols; ++c) {
                std::cout << std::fixed << std::setprecision(4) << v[r * cols + c];
                if (c + 1 < cols) std::cout << ", ";
            }
            std::cout << "]" << (r + 1 < rows ? "\n" : "");
        }
    } else {
        // Generic fallback for higher-rank tensors
        std::cout << "[";
        for (size_t i = 0; i < v.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << v[i];
            if (i + 1 < v.size()) std::cout << ", ";
        }
        std::cout << "]";
    }
}

void Tensor::print(const std::string& name) const {
    std::cout << name << "  shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    print_flat(data, shape);
    std::cout << "\n";
}

void Tensor::print_grad(const std::string& name) const {
    std::cout << name << ".grad  shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 < shape.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    print_flat(grad, shape);
    std::cout << "\n";
}

// ============================================================
//  Static validation
// ============================================================

void Tensor::validate_shapes_equal(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::invalid_argument("Tensor: shapes do not match.");
    }
}
