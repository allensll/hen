#pragma once

#include <iostream>
#include "models.h"

namespace hen {
// @ class SGD
// only fit CNN model
class SGD {
 public:
  SGD(float lr, float momentum);
  void Step(CNN &model);
  void Step(MLP &model);
  void Step(NN1 &model);
 private:
  float lr_;
  float momentum_;
  float gradient_descent(float w, float grad);
  void update_linear(Linear &fc, int batch_size);
  void update_conv(Conv2D &conv, int batch_size);
};

}