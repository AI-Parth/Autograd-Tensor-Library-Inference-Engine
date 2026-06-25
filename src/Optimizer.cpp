#include "../include/Optimizer.h"
#include <stdexcept>
#include <random>
#include <algorithm>

// ============================================================
//  SGD
// ============================================================

SGD::SGD(std::vector<Tensor*> params, float learning_rate)
    : params(std::move(params)), lr(learning_rate) {}

// Apply one gradient-descent step.
// param[i] -= lr * grad[i]
void SGD::step() {
    for (Tensor* p : params) {
        if (p->grad.empty()) continue;
        for (size_t i = 0; i < p->data.size(); ++i)
            p->data[i] -= lr * p->grad[i];
    }
}

// Zero out all parameter gradients.
// Must be called before each forward/backward pass to prevent
// gradient accumulation across iterations.
void SGD::zero_grad() {
    for (Tensor* p : params)
        p->zero_grad();
}

// ============================================================
//  RandomizedCoordinateDescent  (skeleton)
// ============================================================

RandomizedCoordinateDescent::RandomizedCoordinateDescent(
    std::vector<Tensor*> params,
    float                learning_rate,
    float                fraction)
    : params(std::move(params)), lr(learning_rate), fraction(fraction) {}

// TODO: Implement full RCD.
//
// ALGORITHM SKETCH:
//   std::mt19937 rng(std::random_device{}());
//   For each parameter tensor p:
//     size_t total = p->data.size();
//     size_t k     = std::max<size_t>(1, static_cast<size_t>(fraction * total));
//
//     // Create a list of all indices and shuffle to get k random ones
//     std::vector<size_t> indices(total);
//     std::iota(indices.begin(), indices.end(), 0);
//     std::shuffle(indices.begin(), indices.end(), rng);
//     indices.resize(k);
//
//     for (size_t idx : indices)
//       p->data[idx] -= lr * p->grad[idx];
//
// EXTENSIONS (not yet implemented):
//   - Importance sampling proportional to |grad[i]|
//   - KATYUSHA momentum term for faster convergence
//   - Variance-reduced SVRCD variant
void RandomizedCoordinateDescent::step() {
    throw std::runtime_error(
        "RandomizedCoordinateDescent::step() is not yet implemented. "
        "See the comment block above for the implementation sketch.");
}

void RandomizedCoordinateDescent::zero_grad() {
    for (Tensor* p : params)
        p->zero_grad();
}
