#include "../include/Optimizer.h"
#include <random>
#include <algorithm>
#include <numeric>

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

// Randomly select `fraction * total_coords` coordinates per parameter
// and apply the gradient update only to those coordinates.
void RandomizedCoordinateDescent::step() {
    std::mt19937 rng(std::random_device{}());
    for (Tensor* p : params) {
        if (p->grad.empty()) continue;
        size_t total = p->data.size();
        size_t k     = std::max<size_t>(1, static_cast<size_t>(fraction * static_cast<float>(total)));

        std::vector<size_t> indices(total);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(k);

        for (size_t idx : indices)
            p->data[idx] -= lr * p->grad[idx];
    }
}

void RandomizedCoordinateDescent::zero_grad() {
    for (Tensor* p : params)
        p->zero_grad();
}