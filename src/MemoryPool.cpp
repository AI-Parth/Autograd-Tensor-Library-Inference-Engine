#include "../include/MemoryPool.h"
#include "../include/Tensor.h"
#include "../include/Operation.h"

// ============================================================
//  Singleton accessor  (Meyers' Singleton)
// ============================================================
// The first call to instance() constructs the static local object.
// C++11 guarantees this initialisation is thread-safe and happens
// exactly once.  The pool lives until program exit.
MemoryPool& MemoryPool::instance() {
    static MemoryPool pool;
    return pool;
}

// ============================================================
//  registerTensor
// ============================================================
void MemoryPool::registerTensor(Tensor* t) {
    tensors.push_back(t);
}

// ============================================================
//  registerOp
// ============================================================
void MemoryPool::registerOp(Operation* op) {
    ops.push_back(op);
}

// ============================================================
//  cleanup  —  free all tracked objects
// ============================================================
// We delete Operations before Tensors because an Op's destructor
// does not dereference its input/output pointers (it just runs the
// default ~Operation()).  Order is safe either way, but deleting
// Ops first is conceptually cleaner: the graph nodes go away before
// the data nodes.
void MemoryPool::cleanup() {
    for (Operation* op : ops)
        delete op;
    ops.clear();

    for (Tensor* t : tensors)
        delete t;
    tensors.clear();
}