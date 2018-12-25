#pragma once

#include "tensor.h"
#include "nn.h"

namespace hen {
// @ class CNN
// https://github.com/pytorch/examples/blob/master/mnist/main.py
class CNN {
 public:
  CNN(int batch_size);
  void Run(FloatTensor &input, FloatTensor &output);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
//  private:
  // define net
  // FloatTensor input_;
  FloatTensor output_conv1_;
  FloatTensor output_relu1_;
  FloatTensor output_pool1_;
  FloatTensor output_conv2_;
  FloatTensor output_relu2_;
  FloatTensor output_pool2_;
  FloatTensor output_flat1_;
  FloatTensor output_fc1_;
  FloatTensor output_relu3_;
  // FloatTensor output_fc2_;
  // FloatTensor output_;

  Conv2D conv1_;
  Relu relu1_;
  AvgPool2D pool1_;
  Conv2D conv2_;
  Relu relu2_;
  AvgPool2D pool2_;
  Flatten flat1_;
  Linear fc1_;
  Relu relu3_;
  Linear fc2_;
  // Softmax softmax1_;

  int batch_size_;  
};

} // namespace hen