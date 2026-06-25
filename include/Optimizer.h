#pragma once

#include "Tensor.h"
#include <vector>

/**
 * ============================================================
 *  SGD  —  Stochastic Gradient Descent optimizer
 * ============================================================
 *
 * UPDATE RULE (applied once per step):
 *   param[i]  =  param[i]  -  learning_rate * grad[i]
 *
 * INTUITION:
 *   The gradient points in the direction of STEEPEST ASCENT of
 *   the loss.  We move in the OPPOSITE direction (negative gradient)
 *   to decrease the loss.  `learning_rate` (η) controls step size.
 *
 * WORKFLOW:
 *   SGD opt({&W, &B}, 0.01f);
 *
 *   for (int step = 0; step < N; ++step) {
 *       opt.zero_grad();          // 1. clear old gradients
 *       // … forward + backward pass …
 *       opt.step();               // 2. update parameters
 *   }
 * ============================================================
 */
class SGD {
public:
    /**
     * @param params        Pointers to parameter tensors to optimise.
     * @param learning_rate Step size η (default 0.01).
     */
    SGD(std::vector<Tensor*> params, float learning_rate = 0.01f);

    /**
     * Apply one gradient-descent step to every parameter:
     *   param.data[i] -= lr * param.grad[i]
     */
    void step();

    /**
     * Reset all parameter gradients to zero.
     *
     * MUST be called before each forward/backward pass.
     * If omitted, gradients from previous iterations accumulate,
     * producing incorrect (too-large) updates.
     */
    void zero_grad();

private:
    std::vector<Tensor*> params;
    float                lr;
};


/**
 * ============================================================
 *  RandomizedCoordinateDescent  (skeleton — TODO)
 * ============================================================
 *
 * IDEA:
 *   Instead of updating EVERY parameter at once (like SGD), RCD
 *   randomly selects a SUBSET of coordinates each step.
 *
 * POTENTIAL ADVANTAGES:
 *   - Lower cost per iteration when parameters are high-dimensional
 *   - Can exploit gradient sparsity
 *   - Simpler convergence proofs in strongly-convex settings
 *
 * ALGORITHM SKETCH (for step()):
 *   For each parameter tensor p:
 *     total = p->data.size()
 *     k     = max(1, fraction * total)     // number of coords to update
 *     Draw k random indices from [0, total)
 *     For each selected index i:
 *       p->data[i] -= lr * p->grad[i]
 *
 * EXTENSIONS (not yet implemented):
 *   - Importance sampling: pick coordinates proportional to |grad[i]|
 *   - KATYUSHA-style momentum for accelerated convergence
 *   - Variance-reduced SVRCD variant
 * ============================================================
 */
class RandomizedCoordinateDescent {
public:
    /**
     * @param params        Pointers to parameter tensors.
     * @param learning_rate Step size η.
     * @param fraction      Fraction of coordinates to update per step (0 < f ≤ 1).
     */
    RandomizedCoordinateDescent(std::vector<Tensor*> params,
                                float                learning_rate = 0.01f,
                                float                fraction      = 0.5f);

    /**
     * TODO: Randomly select `fraction * total_coords` coordinates
     *       and apply the gradient update to only those.
     */
    void step();

    void zero_grad();

private:
    std::vector<Tensor*> params;
    float                lr;
    float                fraction;
};
