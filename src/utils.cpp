#include "utils.h"

namespace hen {
// random function
void Randomfunction(float data[], int n, int seed) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_real_distribution<float> distribution(-0.1, 0.1);
  for (int i=0; i<n; i++) {
    data[i] = distribution(generator);
    // data[i] = 1;
  }
}

}