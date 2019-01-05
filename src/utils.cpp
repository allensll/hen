#include "utils.h"

namespace hen {
// random function
void Randomfunction(float data[], int n, int seed, float bound) {
  std::default_random_engine generator;
  generator.seed(seed);
  std::uniform_real_distribution<float> distribution(-bound, bound);
  for (int i=0; i<n; i++) {
    data[i] = distribution(generator);
    // data[i] = 0.1;
  }
}

void PrintSample(FloatTensor &tensor) {
  // draw number
  int batch_size = tensor.size_[0];
  int channel = tensor.size_[1];
  int height = tensor.size_[2];
  int weight = tensor.size_[3];
  for (int i=1; i<=batch_size; i++) {
    std::cout << "batch " << i << std::endl;
    for (int c=1; c<=channel; c++) {
      for (int h=1; h<=28; h++) {
        for (int w=1; w<=28; w++) {
          std::cout << std::setfill(' ') << std::setw(4);
          std::cout << (int)tensor.Get({i, 1, h, w});
        }
        std::cout << std::endl;
      }
      std::cout << std::endl << std::endl;
    }
  }
}

int CountCorrect(FloatTensor &output, FloatTensor &target) {
  int batch_size = output.size_[0];
  int cls_num = output.size_[1];
  int correct = 0;
  for (int i=0; i<batch_size; i++) {
    float max_val = 0;
    int max_idx = 0;
    float temp = 0;
    for (int j=0; j<cls_num; j++) {
      temp = output.Get({i+1, j+1});
      if (temp > max_val) {
        max_val = temp;
        max_idx = j;
      }
    }
    if (max_idx == target.Get({i+1})) {
      correct += 1;
    }
  }
  return correct;
}

}