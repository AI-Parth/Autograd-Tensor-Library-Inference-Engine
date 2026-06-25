#include "../include/GraphEngine.h"
#include <stdexcept>
#include <algorithm>

// ============================================================
//  backward()  —  entry point for the full backward pass
// ============================================================
void GraphEngine::backward(Tensor* loss) {
    if (loss->size() != 1)
        throw std::runtime_error(
            "GraphEngine::backward: loss must be a scalar tensor (size == 1).");

    // ---- Step 1: Seed gradient ----
    // By definition, dL/dL = 1.0.  This is the "seed" gradient that
    // starts the chain-rule traversal.
    loss->grad.assign(1, 1.0f);

    // ---- Step 2: Collect operations in topological order ----
    // We walk the graph from the loss node backwards, visiting each
    // Operation that contributed to the loss.
    std::vector<Operation*>           order;
    std::unordered_set<Operation*>    visited;

    if (loss->creator != nullptr)
        build_topo(loss->creator, order, visited);

    // ---- Step 3: Execute backward in REVERSE topological order ----
    //
    // `order` is in topological order (inputs → output).
    // We reverse it so we go output → inputs, which means each Op
    // receives its output gradient BEFORE its own backward() runs.
    //
    // Example for  Z = ReLU(Add(MatMul(X,W), B)):
    //   topological order:  [MatMul, Add, ReLU, MSELoss]
    //   reversed:           [MSELoss, ReLU, Add, MatMul]
    //   MSELoss fills ReLU-output's grad → ReLU fills Add-output's grad
    //   → Add fills MatMul-output's grad + B's grad
    //   → MatMul fills W's grad
    for (auto it = order.rbegin(); it != order.rend(); ++it)
        (*it)->backward();
}

// ============================================================
//  build_topo()  —  DFS post-order traversal
// ============================================================
// We visit every ancestor of `op` recursively BEFORE appending `op`
// itself.  This yields topological order:
//   parents always appear before the ops that consume their outputs.
void GraphEngine::build_topo(Operation*                      op,
                              std::vector<Operation*>&        order,
                              std::unordered_set<Operation*>& visited)
{
    // If we have already visited this op (possible in diamond-shaped
    // graphs where one tensor feeds two downstream ops), skip it.
    if (visited.count(op)) return;
    visited.insert(op);

    // Recursively visit the creator of each input tensor.
    for (Tensor* inp : op->inputs) {
        if (inp->creator != nullptr)
            build_topo(inp->creator, order, visited);
    }

    // All ancestors have been pushed; now push the current op.
    order.push_back(op);
}
