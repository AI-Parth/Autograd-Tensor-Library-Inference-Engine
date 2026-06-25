#pragma once

#include "Module.h"

namespace flash {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);

    ::Tensor* forward(::Tensor* input) override;
    std::vector<::Tensor*> parameters() override;

    ::Tensor W;
    ::Tensor B;

private:
    bool use_bias;
};

}  // namespace flash
