#include "../../include/nn/ReLULayer.h"
#include "../../include/autograd.h"

namespace flash {

::Tensor* ReLULayer::forward(::Tensor* input) {
    return ::relu(input);
}

}  // namespace flash
