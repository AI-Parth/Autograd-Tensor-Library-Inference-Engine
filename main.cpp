/**
 * ============================================================
 *  main.cpp  —  End-to-end demonstration of the flash API
 * ============================================================
 *
 * This demo mirrors the module / optimizer / training-loop style
 * used by popular Python deep-learning libraries, but through the
 * `flash::` API exposed by `include/flash.h`.
 *
 * MODEL:
 *   flash::Sequential(
 *     flash::Linear(2, 8),
 *     flash::ReLULayer(),
 *     flash::Linear(8, 3)
 *   )
 * ============================================================
 */

#include <iostream>
#include <iomanip>
#include <memory>

#include "include/flash.h"

int main() {
    std::cout << "============================================================\n";
    std::cout << "  flash  —  Sequential module demo\n";
    std::cout << "============================================================\n\n";

    flash::Tensor X(
        std::vector<float>{ 1.0f, 2.0f,   // sample 0
                            3.0f, 4.0f,   // sample 1
                            5.0f, 6.0f,   // sample 2
                            7.0f, 8.0f }, // sample 3
        std::vector<size_t>{4, 2},
        /*requires_grad=*/ false
    );

    flash::Tensor Y_target(
        std::vector<float>{  1.0f,  0.0f, -1.0f,
                             2.0f, -1.0f,  0.0f,
                            -1.0f,  2.0f,  1.0f,
                             0.0f,  1.0f, -2.0f },
        std::vector<size_t>{4, 3},
        /*requires_grad=*/ false
    );

    flash::Sequential model;
    model.add(std::make_unique<flash::Linear>(2, 8));
    model.add(std::make_unique<flash::ReLULayer>());
    model.add(std::make_unique<flash::Linear>(8, 3));

    std::cout << "---- Dataset ----\n";
    X.print("X  [4×2 input]");
    Y_target.print("Y_target  [4×3 targets]");
    std::cout << "\n";

    flash::Adam optimizer(model.parameters(), /*learning_rate=*/ 0.001f);
    constexpr int NUM_STEPS = 100;

    std::cout << "---- Training (" << NUM_STEPS << " steps) ----\n";

    for (int step = 0; step < NUM_STEPS; ++step) {
        optimizer.zero_grad();

        flash::Tensor* pred = model.forward(&X);
        flash::Tensor* loss = flash::mse_loss(pred, &Y_target);

        if (step == 0 || (step + 1) % 20 == 0) {
            std::cout << "  Step " << std::setw(2) << (step + 1)
                      << "  |  Loss = "
                      << std::fixed << std::setprecision(6)
                      << loss->data[0] << "\n";
        }

        flash::GraphEngine::backward(loss);
        optimizer.step();
        flash::MemoryPool::instance().cleanup();
    }

    std::cout << "\n============================================================\n";
    std::cout << "  Final Parameters After " << NUM_STEPS << " Training Steps\n";
    std::cout << "============================================================\n";

    std::vector<flash::Tensor*> params = model.parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i]->print("parameter[" + std::to_string(i) + "]");
    }

    std::cout << "\n---- Final Predictions (no gradient tracking) ----\n";
    flash::Tensor* pred_final = model.forward(&X);
    pred_final->print("Y_hat (final predictions)");
    Y_target.print("Y_target (ground truth)");
    flash::MemoryPool::instance().cleanup();

    std::cout << "\nDone.\n";
    return 0;
}