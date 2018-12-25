#include "optim.h"

namespace hen {
// @ class SGD
SGD::SGD(float lr, float momentum) {
  lr_ = lr;
  momentum_ = momentum; 
}

void SGD::Step(CNN &model) {
  // FloatTensor output_conv1_;
  // FloatTensor output_relu1_;
  // FloatTensor output_pool1_;
  // FloatTensor output_conv2_;
  // FloatTensor output_relu2_;
  // FloatTensor output_pool2_;
  // FloatTensor output_flat1_;
  // FloatTensor output_fc1_;
  // FloatTensor output_relu3_;
  // // FloatTensor output_fc2_;
  // // FloatTensor output_;

  // Conv2D conv1_;
  // Relu relu1_;
  // AvgPool2D pool1_;
  // Conv2D conv2_;
  // Relu relu2_;
  // AvgPool2D pool2_;
  // Flatten flat1_;
  // Linear fc1_;
  // Relu relu3_;
  // Linear fc2_;
  int batch_size = model.batch_size_;

  update_linear(model.fc2_, batch_size);
  update_linear(model.fc1_, batch_size);
  update_conv(model.conv2_, batch_size);
  update_conv(model.conv1_, batch_size);
}

float SGD::gradient_descent(float w, float grad) {
  return w - lr_ * grad;
}

void SGD::update_linear(Linear &fc, int batch_size) {
  for (int o=1; o<=fc.output_n_; o++) {
    for (int i=1; i<fc.input_n_; i++) {
      float grad = 0;
      for (int batch=0; batch<batch_size; batch++) {
        grad += fc.weights_[batch].GetGrad(o, i);
      }
      grad /= batch_size;
      float w = fc.weights_[0].Get(o, i);
      w = gradient_descent(w, grad);
      for (int batch=0; batch<=batch_size; batch++) {
        fc.weights_[batch].Set(o, i, w);
      }
    }
    float grad = 0;
    for (int batch=0; batch<batch_size; batch++) {
      grad += fc.weights_[batch].bias_d_[o-1];
    }
    grad /= batch_size;
    float w = fc.weights_[0].bias_[o-1];
    w = gradient_descent(w, grad);
    for (int batch=0; batch<=batch_size; batch++) {
      fc.weights_[batch].bias_[o-1] = w;
    }
  }
}

void SGD::update_conv(Conv2D &conv, int batch_size) {
  int num = conv.kernel_size_ * conv.kernel_size_;
  for (int ch=0; ch<conv.output_channel_; ch++) {
    for (int n=0; n<num; n++) {
      float grad = 0;
      for (int batch=0; batch<batch_size; batch++) {
        grad += conv.kernels_[batch*batch_size+ch].weight_d_[ch*num+n];
      }
      grad /= batch_size;
      float w = conv.kernels_[ch].weight_[ch*num+n];
      w = gradient_descent(w, grad);
      for (int batch=0; batch<batch_size; batch++) {
        conv.kernels_[batch*batch_size+ch].weight_[ch*num+n] = w;
      }
    }
    float grad = 0;
    for (int batch=0; batch<batch_size; batch++) {
      grad += conv.kernels_[batch*batch_size+ch].bias_d_;
    }
    float w = conv.kernels_[ch].bias_;
    w /= batch_size;
    for (int batch=0; batch<batch_size; batch++) {
      conv.kernels_[batch*batch_size+ch].bias_ = w;
    }
  }
}

}