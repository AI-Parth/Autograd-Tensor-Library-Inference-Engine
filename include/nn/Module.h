#pragma once

#include "../Tensor.h"
#include <vector>

namespace flash {

class Module {
public:
    virtual ::Tensor* forward(::Tensor* input) = 0;
    virtual std::vector<::Tensor*> parameters() = 0;

    void zero_grad() {
        for (::Tensor* parameter : parameters()) {
            if (parameter != nullptr) {
                parameter->zero_grad();
            }
        }
    }

    virtual ~Module() = default;
};

}  // namespace flash
