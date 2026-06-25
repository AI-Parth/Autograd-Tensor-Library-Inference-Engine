#pragma once

#include "Module.h"
#include <memory>
#include <vector>

namespace flash {

class Sequential : public Module {
public:
    void add(std::unique_ptr<Module> layer);
    ::Tensor* forward(::Tensor* input) override;
    std::vector<::Tensor*> parameters() override;

private:
    std::vector<std::unique_ptr<Module>> layers;
};

}  // namespace flash
