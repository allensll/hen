#pragma once

#include <cmath>
#include <iostream>
#include <seal/seal.h>
#include "tensor.h"
#include "utils.h"

// all Tensor size is {batch_size, channel_size, height, weight}

namespace hen {
// @ class Weight
// used in Linear layer
class Weight {
 public:
  int input_n_;
  int output_n_;
  int weight_n_;
  int bias_n_;
  float* weight_;   // { output_n_, input_n_ }
  float* weight_d_;
  float* bias_;     // { output_n_ }
  float* bias_d_;
  Weight();
  Weight(int input, int output, int seed);
  Weight & operator=(const Weight &assign);
  ~Weight();
  float Get(int i, int j);
  float GetGrad(int i, int j);
  void Set(int i, int j, float value);
  void SetGrad(int i, int j, float value);
  void Add(int i, int j, float value);
  void AddGrad(int i, int j, float value);
  void ZeroGrad();
};

// @ class Kernel
class Kernel {
 public:
  int channel_;
  int kernel_size_;
  int weight_n_;
  float* weight_;  // { channel, kernel_size, kernel_size }
  float* weight_d_;
  float bias_;
  float bias_d_;
  Kernel();
  Kernel(int channel, int kernel_size_, int seed);
  Kernel & operator=(const Kernel &assign);
  ~Kernel();
  void ZeroGrad();
 private:
};

// @ class Linear
// input { batch_size, input_size }
// output { batch_size, output_size }
class Linear {
 public:
  // FloatTensor &input_;
  // FloatTensor &output_;
  int batch_size_;
  int input_n_;
  int output_n_;
  Weight* weights_; // { batch_size }
  Linear(int batch_size, int input_n, int output_n);
  // Linear(FloatTensor &input, FloatTensor &output);
  // Linear(seal::Evaluator* evaluator, FloatTensor &input, FloatTensor &output);
  ~Linear();
  void ZeroGrad(FloatTensor &input);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
 private:
  seal::Evaluator* evaluator_;
};

// @ class Conv2D
class Conv2D {
 public:
  // FloatTensor input_;
  // FloatTensor output_;
  int batch_size_;
  int input_channel_;
  // int input_h_;
  // int input_w_;
  int output_channel_;
  // int output_h_;
  // int output_w_;
  int kernel_size_;
  int stride_;
  Kernel* kernels_;  // { batch_size, output_channel }
  int kernel_n_;   // batch_size_ * output_channel
  Conv2D(int batch_size, int input_channel, int output_channel, int kernel_size, int stride);
  // Conv2D(FloatTensor &input, FloatTensor &output, int kernel_size, int stride);
  // Conv2D(seal::Evaluator, FloatTensor &input, FloatTensor &output);
  ~Conv2D();
  void ZeroGrad(FloatTensor &input);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
  void tensor2kernels(FloatTensor &input, Kernel* kernels, int stride);         // ????? const
//  private:
  float convolution(const Kernel &kernel, const float* const* const data);
  void expandmap(FloatTensor &input, FloatTensor &output, int stride=1);
  void transkernels(const Kernel* const k_input, Kernel* k_output, int k_input_n);
  // seal::Evaluator &evaluator_;
};

// @ class AvgPool2D
class AvgPool2D {
 public:
  // FloatTensor input_;
  // FloatTensor output_;
  // int batch_size_;
  // int channel_;
  // int input_w_;
  // int input_h_;
  // int output_w_;
  // int output_h_;
  int kernel_size_;
  int stride_;       // stride_ = kernel_size_
  AvgPool2D(int kernel_size, int stride);
  // AvgPool2D(FloatTensor &input, FloatTensor &output, int kernel_size);
  float pool(float kernel[], int n);
  void ZeroGrad(FloatTensor &input);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
  // AvgPool2D(seal::Evaluator, FloatTensor &input, FloatTensor &output_);
 private:
  // seal::Evaluator* evaluator_;
};

// @ class Flatten
class Flatten {
 public:
  Flatten();
  void ZeroGrad(FloatTensor &input);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
};



// @ class MaxPool2D



// @ class Relu
// Tensor size is {batch_size, channel_size, height, weight}
class Relu {
 public:
  // FloatTensor &input_;
  // FloatTensor &output_;
  int batch_size_;
  Relu();
  // Relu(FloatTensor &input, FloatTensor &output);
  inline float relu(float x);
  inline float relu_d(float x);
  void ZeroGrad(FloatTensor &input);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
};

// @ class Sigmoid
// Tensor size is {batch_size, channel_size, height, weight}
class Sigmoid {
 public:
  FloatTensor input_;
  FloatTensor output_;
  int batch_size_;
  Sigmoid(FloatTensor &input, FloatTensor &output);
  inline float sigmoid(float x);
  inline float sigmoid_d(float x);
  void Forward();
  void Backward();
};

// @ class Softmax
// Tensor size is {batch_size, class_num}
class Softmax {
 public:
  // FloatTensor input_;
  // FloatTensor output_;
  // int batch_size_;
  // int class_num_;
  Softmax();
  // Softmax(FloatTensor &input, FloatTensor &output);
  inline void softmax(float x[], int n);
  inline void softmax_d(float x[], int n);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
};

// @ class LogSoftmax
// Tensor size is { batch_size, class_num }
class LogSoftmax {
 public:
  LogSoftmax();
  inline void log_softmax(float x[], int n);
  inline void log_softmax_d(float x[], int n);
  void Forward(FloatTensor &input, FloatTensor &output);
  void Backward(FloatTensor &input, FloatTensor &output);
};


// @ class NLLLoss
class NLLLoss {
 public:
  NLLLoss();
  float Loss(FloatTensor &input, IntTensor target);
};

// @ class CrossEntLoss
class CrossEntropyLoss {
 public:
  CrossEntropyLoss();
  // input { batch_size, class_num }, target { batch_size } label start from 0
  float Loss(FloatTensor &input, FloatTensor &target);
  void Forward();
  void Backward(FloatTensor &input, FloatTensor &output);
  void ZeroGrad(FloatTensor &input);
 private:
  // void softmax(FloatTensor &input, FloatTensor &output);
};

// @ function Pad()
void ZeroPading2D(const FloatTensor &input, FloatTensor&output, int stride=0);



} // namespace hen