#pragma once

#include <seal/seal.h>

namespace hen {
// @class Config
class Pack {
  public:
    std::shared_ptr<seal::SEALContext> context_;
    seal::PublicKey public_key_;
    Pack(std::shared_ptr<seal::SEALContext> context, seal::PublicKey public_key);
};
} // namespace hen