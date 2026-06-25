#include "../../include/nn/Linear.h"
#include "../../include/autograd.h"

#include <random>

namespace flash {

Linear::Linear(size_t in_features, size_t out_features, bool bias)
    : W(std::vector<size_t>{in_features, out_features}, 0.0f, true)
    , B(std::vector<size_t>{out_features}, 0.0f, bias)
    , use_bias(bias)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (float& weight : W.data) {
        weight = dist(rng);
    }
}

::Tensor* Linear::forward(::Tensor* input) {
    ::Tensor* output = ::matmul(input, &W);
    if (use_bias) {
        output = ::add(output, &B);
    }
    return output;
}

std::vector<::Tensor*> Linear::parameters() {
    if (use_bias) {
        return {&W, &B};
    }
    return {&W};
}

}  // namespace flash
