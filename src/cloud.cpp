#include <seal/seal.h>
#include "cloud.h"

namespace hen {
// @class Cloud
Cloud::Cloud(Pack pack) :
  context_(pack.context_),
  public_key_(pack.public_key_),
  evaluator_(context_)
{
  // Cloud(pack.context_, pack.public_key_);
}

// Cloud::Cloud(std::shared_ptr<seal::SEALContext> context, seal::PublicKey public_key) {
//   context_ = context;
//   public_key = public_key; // no use
//   evaluator_(context_);
//   // evaluator_ = temp;
//   // evaluator_ = new seal::Evaluator(context_);
// }

std::vector<seal::Ciphertext> Cloud::Compute(std::vector<seal::Ciphertext> input) {
  std::vector<seal::Ciphertext> output;
  seal::Ciphertext res;
  evaluator_.add(input[0], input[1], res);
  output.insert(output.end(), res);
  return output;
}

} // namespace hen