#include "../../include/nn/Sequential.h"

namespace flash {

void Sequential::add(std::unique_ptr<Module> layer) {
    layers.push_back(std::move(layer));
}

::Tensor* Sequential::forward(::Tensor* input) {
    ::Tensor* output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

std::vector<::Tensor*> Sequential::parameters() {
    std::vector<::Tensor*> params;
    for (const auto& layer : layers) {
        std::vector<::Tensor*> layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

}  // namespace flash
