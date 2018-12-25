#include "utils.h"

namespace hen {
// random function
float Randomfunction() {
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-2, 2);

  return distribution(generator);
}

}