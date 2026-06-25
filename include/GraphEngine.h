#pragma once

#include "Tensor.h"
#include "Operation.h"
#include <vector>
#include <unordered_set>

/**
 * ============================================================
 *  GraphEngine  —  Computation-graph executor
 * ============================================================
 *
 * RESPONSIBILITIES:
 *   1. Traverse the DAG (Directed Acyclic Graph) backwards from
 *      the loss tensor to collect all connected Operation nodes.
 *   2. Sort them in TOPOLOGICAL ORDER (parents before children).
 *   3. Execute the backward pass in REVERSE topological order so
 *      that gradients flow correctly from loss → inputs.
 *
 * ---- WHY REVERSE TOPOLOGICAL ORDER? -----------------------
 *   Each Operation's backward() reads `output->grad`, which must
 *   be fully accumulated BEFORE backward() is called on that Op.
 *   "Fully accumulated" means every downstream Op that used this
 *   output has already run its own backward() and contributed its
 *   gradient share.
 *
 *   Topological order guarantees: if Op_B uses the output of Op_A,
 *   then Op_A appears BEFORE Op_B in topological order.
 *   Therefore, in REVERSED order, Op_B runs before Op_A, which
 *   means Op_B will have already filled Op_A's output gradient
 *   by the time Op_A's backward() is called.  ✓
 *
 * ---- TOPOLOGICAL SORT (DFS post-order) ---------------------
 *   Algorithm:
 *     1. Recursively visit every Op that produced the current Op's
 *        inputs (i.e., go deeper into the graph).
 *     2. After ALL ancestors have been visited, append the current
 *        Op to the `order` list.
 *
 *   This "visit children first, then append self" rule naturally
 *   yields topological order:  Op_A is added before Op_B whenever
 *   Op_B depends on Op_A's output.
 *
 *   A `visited` set prevents revisiting nodes in diamond-shaped
 *   graphs (where one tensor feeds into two different Ops).
 * ============================================================
 */
class GraphEngine {
public:
    /**
     * Run the full backward pass starting from `loss`.
     *
     * PRECONDITION: loss->size() == 1  (must be a scalar tensor).
     *
     * The function:
     *   1. Seeds loss->grad = {1.0f}   (dL/dL = 1 by definition)
     *   2. Builds the topological order via DFS
     *   3. Calls backward() on each Op in reverse topological order
     */
    static void backward(Tensor* loss);

private:
    /**
     * DFS post-order traversal to collect operations in topological order.
     *
     * @param op      Current operation node being visited
     * @param order   Output list (ops are appended in post-order)
     * @param visited Set of already-visited ops (prevents duplicates)
     */
    static void build_topo(Operation*                      op,
                           std::vector<Operation*>&        order,
                           std::unordered_set<Operation*>& visited);
};
