#pragma once

#include "Module.h"

namespace flash {

class ReLULayer : public Module {
public:
    ::Tensor* forward(::Tensor* input) override;
    std::vector<::Tensor*> parameters() override { return {}; }
};

}  // namespace flash
