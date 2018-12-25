#pragma once

#include <seal/seal.h>
#include "pack.h"

namespace hen {
// @class Cloud
class Cloud {
  public:
    Cloud(Pack pack);
    // Cloud(std::shared_ptr<seal::SEALContext> context, seal::PublicKey public_key);
    // ~Cloud();
    std::vector<seal::Ciphertext> Compute(std::vector<seal::Ciphertext> input);
  private:
    std::shared_ptr<seal::SEALContext> context_;
    seal::PublicKey public_key_;
    seal::Evaluator evaluator_;

};
} // namespace hen