#include "../include/Optimizer.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

Optimizer::Optimizer(std::vector<Tensor*> params, float learning_rate)
    : params(std::move(params)), lr(learning_rate) {}

void Optimizer::zero_grad() {
    for (Tensor* p : params) {
        if (p != nullptr) {
            p->zero_grad();
        }
    }
}

// ============================================================
//  SGD
// ============================================================

SGD::SGD(std::vector<Tensor*> params, float learning_rate)
    : Optimizer(std::move(params), learning_rate) {}

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
    Optimizer::zero_grad();
}

// ============================================================
//  RandomizedCoordinateDescent  (skeleton)
// ============================================================

RandomizedCoordinateDescent::RandomizedCoordinateDescent(
    std::vector<Tensor*> params,
    float                learning_rate,
    float                fraction)
    : Optimizer(std::move(params), learning_rate), fraction(fraction) {
    if (fraction <= 0.0f || fraction > 1.0f) {
        throw std::invalid_argument(
            "RandomizedCoordinateDescent: fraction must be in (0, 1].");
    }
}

// Randomly select `fraction * total_coords` coordinates per parameter
// and apply the gradient update only to those coordinates.
void RandomizedCoordinateDescent::step() {
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
    Optimizer::zero_grad();
}

Adam::Adam(std::vector<Tensor*> params,
           float                learning_rate,
           float                beta1,
           float                beta2,
           float                eps)
    : Optimizer(std::move(params), learning_rate)
    , beta1(beta1)
    , beta2(beta2)
    , eps(eps)
    , timestep(0)
{
    if (beta1 <= 0.0f || beta1 >= 1.0f || beta2 <= 0.0f || beta2 >= 1.0f) {
        throw std::invalid_argument("Adam: beta values must be in (0, 1).");
    }
    if (eps <= 0.0f) {
        throw std::invalid_argument("Adam: eps must be positive.");
    }

    first_moment.reserve(this->params.size());
    second_moment.reserve(this->params.size());

    for (Tensor* p : this->params) {
        const size_t size = (p == nullptr) ? 0 : p->data.size();
        first_moment.emplace_back(size, 0.0f);
        second_moment.emplace_back(size, 0.0f);
    }
}

void Adam::step() {
    ++timestep;
    const float beta1_correction = 1.0f - std::pow(beta1, static_cast<float>(timestep));
    const float beta2_correction = 1.0f - std::pow(beta2, static_cast<float>(timestep));

    for (size_t param_index = 0; param_index < params.size(); ++param_index) {
        Tensor* p = params[param_index];
        if (p == nullptr || p->grad.empty()) continue;

        for (size_t i = 0; i < p->data.size(); ++i) {
            const float grad = p->grad[i];
            first_moment[param_index][i] =
                beta1 * first_moment[param_index][i] + (1.0f - beta1) * grad;
            second_moment[param_index][i] =
                beta2 * second_moment[param_index][i] + (1.0f - beta2) * grad * grad;

            const float m_hat = first_moment[param_index][i] / beta1_correction;
            const float v_hat = second_moment[param_index][i] / beta2_correction;

            p->data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

void Adam::zero_grad() {
    Optimizer::zero_grad();
}