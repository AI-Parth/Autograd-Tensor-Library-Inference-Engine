/**
 * ============================================================
 *  main.cpp  —  End-to-end demonstration of the Autograd Engine
 * ============================================================
 *
 * MODEL: Single linear layer with ReLU activation
 *   Y_hat = ReLU( X * W + B )
 *
 *   X  : [4 samples × 2 features]     (input data,  no gradient)
 *   W  : [2 features × 3 outputs]     (weight matrix, learnable)
 *   B  : [3 outputs]                  (bias vector,   learnable, broadcast)
 *
 * LOSS: Mean Squared Error against a target matrix
 *   MSE = (1/N) * Σ_i (Y_hat_i - Y_target_i)^2
 *
 * OPTIMIZER: SGD with learning rate 0.001
 *
 * TRAINING LOOP:
 *   for each step:
 *     1. zero_grad()          — clear accumulated gradients
 *     2. forward pass         — compute Y_hat and loss
 *     3. backward pass        — propagate gradients through the graph
 *     4. optimizer.step()     — update W and B
 *     5. cleanup()            — free intermediate tensors/ops
 *
 * COMPILATION:
 *   g++ -std=c++20 -Wall -O2 -o autograd_engine \
 *       src/Tensor.cpp src/GraphEngine.cpp src/MemoryPool.cpp \
 *       src/Optimizer.cpp src/ops/Add.cpp src/ops/MatMul.cpp \
 *       src/ops/ReLU.cpp src/ops/MSELoss.cpp main.cpp
 * ============================================================
 */

#include <iostream>
#include <iomanip>
#include "include/autograd.h"

int main() {
    std::cout << "============================================================\n";
    std::cout << "  Autograd Tensor Library & Inference Engine  —  Demo\n";
    std::cout << "============================================================\n\n";

    // ----------------------------------------------------------
    //  1. Input data  X : [4 samples × 2 features]
    //     Each row is one training sample.
    //     No gradient needed — X is fixed data, not a parameter.
    // ----------------------------------------------------------
    Tensor X(
        std::vector<float>{ 1.0f, 2.0f,   // sample 0
                            3.0f, 4.0f,   // sample 1
                            5.0f, 6.0f,   // sample 2
                            7.0f, 8.0f }, // sample 3
        std::vector<size_t>{4, 2},
        /*requires_grad=*/ false
    );

    // ----------------------------------------------------------
    //  2. Weight matrix  W : [2 features × 3 outputs]
    //     Small random-ish initial values to avoid immediate saturation.
    //     requires_grad = true  →  gradients will be computed for W.
    // ----------------------------------------------------------
    Tensor W(
        std::vector<float>{  0.10f, -0.20f,  0.30f,
                            -0.10f,  0.40f, -0.20f },
        std::vector<size_t>{2, 3},
        /*requires_grad=*/ true
    );

    // ----------------------------------------------------------
    //  3. Bias vector  B : [3 outputs]
    //     Will be broadcast across the batch dimension during add().
    //     requires_grad = true  →  gradients will be computed for B.
    // ----------------------------------------------------------
    Tensor B(
        std::vector<float>{ 0.10f, 0.00f, -0.10f },
        std::vector<size_t>{3},
        /*requires_grad=*/ true
    );

    // ----------------------------------------------------------
    //  4. Target matrix  Y_target : [4 samples × 3 outputs]
    //     These are the ground-truth labels we want to predict.
    //     No gradient — targets are constants.
    // ----------------------------------------------------------
    Tensor Y_target(
        std::vector<float>{  1.0f,  0.0f, -1.0f,
                             2.0f, -1.0f,  0.0f,
                            -1.0f,  2.0f,  1.0f,
                             0.0f,  1.0f, -2.0f },
        std::vector<size_t>{4, 3},
        /*requires_grad=*/ false
    );

    // ----------------------------------------------------------
    //  Print initial state
    // ----------------------------------------------------------
    std::cout << "---- Initial Parameters ----\n";
    X.print("X  [4×2 input]");
    W.print("W  [2×3 weights]");
    B.print("B  [3   bias]");
    Y_target.print("Y_target  [4×3 targets]");
    std::cout << "\n";

    // ----------------------------------------------------------
    //  5. Optimizer: SGD with learning rate η = 0.001
    // ----------------------------------------------------------
    SGD optimizer({&W, &B}, /*learning_rate=*/ 0.001f);

    // ----------------------------------------------------------
    //  6. Training loop  (20 gradient-descent steps)
    // ----------------------------------------------------------
    constexpr int NUM_STEPS = 20;

    std::cout << "---- Training (" << NUM_STEPS << " steps) ----\n";

    for (int step = 0; step < NUM_STEPS; ++step) {

        // ---- 6a. Zero gradients ----
        // Clear W.grad and B.grad from the previous iteration.
        // Without this, gradients from multiple steps would sum up,
        // causing incorrect (ever-growing) parameter updates.
        optimizer.zero_grad();

        // ---- 6b. Forward pass ----

        // Step 1: Z = X * W   →  [4,2] × [2,3] = [4,3]
        // Each row of Z is the linear projection of one sample.
        Tensor* Z = matmul(&X, &W);

        // Step 2: ZB = Z + B   →  [4,3] + [3] = [4,3]  (broadcasting B)
        // Adds the bias B to every row of Z.
        Tensor* ZB = add(Z, &B);

        // Step 3: A = ReLU(ZB)   →  [4,3]
        // Non-linear activation: negative values are clamped to 0.
        Tensor* A = relu(ZB);

        // Step 4: loss = MSE(A, Y_target)   →  scalar [1]
        // Measures how far our predictions are from the targets.
        Tensor* loss = mse_loss(A, &Y_target);

        // ---- 6c. Print progress ----
        if (step == 0 || (step + 1) % 5 == 0) {
            std::cout << "  Step " << std::setw(2) << (step + 1)
                      << "  |  Loss = "
                      << std::fixed << std::setprecision(6)
                      << loss->data[0] << "\n";
        }

        // ---- 6d. Backward pass ----
        // GraphEngine traverses the computation graph in reverse
        // topological order and calls backward() on each Op, applying
        // the chain rule to propagate gradients back to W and B.
        GraphEngine::backward(loss);

        // ---- 6e. Parameter update ----
        // W and B are moved in the direction that reduces the loss:
        //   W -= 0.001 * W.grad
        //   B -= 0.001 * B.grad
        optimizer.step();

        // ---- 6f. Free intermediate graph nodes ----
        // Z, ZB, A, loss (and the four Op objects) are heap-allocated
        // intermediates registered with MemoryPool.  We free them here.
        // W, B, X, Y_target are NOT in the pool — they survive.
        MemoryPool::instance().cleanup();
    }

    // ----------------------------------------------------------
    //  7. Final results
    // ----------------------------------------------------------
    std::cout << "\n============================================================\n";
    std::cout << "  Final Parameters After " << NUM_STEPS << " Training Steps\n";
    std::cout << "============================================================\n";
    W.print("W  (updated weights)");
    B.print("B  (updated bias)");

    std::cout << "\n---- Last-step Gradients (after final iteration) ----\n";
    std::cout << "(These are zero because zero_grad() was called before the\n"
              << " last step, and cleanup() freed the graph — for illustration\n"
              << " remove the final zero_grad() call to see non-zero values.)\n\n";
    W.print_grad("W");
    B.print_grad("B");

    // ----------------------------------------------------------
    //  8. One final forward pass to show predictions
    // ----------------------------------------------------------
    std::cout << "\n---- Final Predictions (no gradient tracking) ----\n";
    Tensor* Z_final    = matmul(&X, &W);
    Tensor* ZB_final   = add(Z_final, &B);
    Tensor* pred_final = relu(ZB_final);
    pred_final->print("Y_hat (final predictions)");
    Y_target.print("Y_target (ground truth)");
    MemoryPool::instance().cleanup();

    std::cout << "\nDone.\n";
    return 0;
}