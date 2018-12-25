#include <seal/seal.h>
#include "pack.h"

namespace hen {
// @class Config
Pack::Pack(std::shared_ptr<seal::SEALContext> context, seal::PublicKey pkey) {
  context_ = context;
  public_key_ = pkey;
}
} // namespace hen