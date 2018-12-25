#include "nn.h"

namespace hen {
// @ class Weight
Weight::Weight() {
  weight_ = new float[1];
  weight_d_ = new float[1];
  bias_ = new float[1];
  bias_d_ = new float[1];
}

Weight::Weight(int input, int output) {
  input_n_ = input;
  output_n_ = output;
  weight_n_ = input * output;
  bias_n_ = output;
  weight_ = new float[weight_n_];
  weight_d_ = new float[weight_n_];
  bias_ = new float[bias_n_];
  bias_d_ = new float[bias_n_];
  for (int i=0; i<weight_n_; i++) {
    weight_[i] = Randomfunction();
    weight_d_[i] = 0;
  }
  for (int i=0; i<bias_n_; i++) {
    bias_[i] = Randomfunction();
    bias_d_[i] = 0;
  }
}
Weight & Weight::operator=(const Weight &assign) {
  if (this == &assign)
    return *this;
  delete [] weight_;
  delete [] weight_d_;
  delete [] bias_;
  delete [] bias_d_;

  input_n_ = assign.input_n_;
  output_n_ = assign.output_n_;
  weight_n_ = input_n_ * output_n_;
  bias_n_ = output_n_;
  weight_ = new float[weight_n_];
  weight_d_ = new float[weight_n_];
  bias_ = new float[bias_n_];
  bias_d_ = new float[bias_n_];
  for (int i=0; i<weight_n_; i++) {
    weight_[i] = assign.weight_[i];
    weight_d_[i] = assign.weight_d_[i];
  }
  for (int i=0; i<bias_n_; i++) {
    bias_[i] = assign.bias_[i];
    bias_d_[i] = assign.bias_d_[i];
  }
  return *this;
}

Weight::~Weight() {
  delete [] weight_;
  delete [] weight_d_;
  delete [] bias_;
  delete [] bias_d_;
}
float Weight::Get(int i, int j) {
  if (i>output_n_ || j>input_n_) {
    std::cout << "Weight Get error" << std::endl;
    return 0;
  }
  int idx = (i-1) * input_n_ + (j-1);
  return weight_[idx];
}
float Weight::GetGrad(int i, int j) {
  if (i>output_n_ || j>input_n_) {
    std::cout << "Weight GetGrad error" << std::endl;
    return 0;
  }
  int idx = (i-1) * input_n_ + (j-1);
  return weight_d_[idx];
}
void Weight::Set(int i, int j, float value) {
  if (i>output_n_ || j>input_n_) {
    std::cout << "Weight Set error" << std::endl;
    return;
  }
  int idx = (i-1) * input_n_ + (j-1);
  weight_[idx] = value;
}
void Weight::SetGrad(int i, int j, float value) {
  if (i>output_n_ || j>input_n_) {
    std::cout << "Weight SetGrad error" << std::endl;
    return;
  }
  int idx = (i-1) * input_n_ + (j-1);
  weight_d_[idx] = value;
}
void Weight::Add(int i, int j, float value) {
  if (i>output_n_ || j>input_n_) {
    std::cout << "Weight Add error" << std::endl;
    return;
  }
  int idx = (i-1) * input_n_ + (j-1);
  weight_[idx] += value;
}
void Weight::AddGrad(int i, int j, float value) {
  if (i>output_n_ || j>input_n_) {
    std::cout << "Weight AddGrad error" << std::endl;
    return;
  }
  int idx = (i-1) * input_n_ + (j-1);
  weight_d_[idx] += value;
}

void Weight::ZeroGrad() {
  for (int i=0; i<weight_n_; i++) {
    weight_d_ = 0;
  }
  for (int i=0; i<bias_n_; i++) {
    bias_d_[i] = 0;
  }
}


// @ class Kernel
Kernel::Kernel() {
  weight_ = new float[0];
  weight_d_ = new float[0];
}

Kernel::Kernel(int channel, int kernel_size) {
  kernel_size_ = kernel_size;
  weight_n_ = channel * kernel_size * kernel_size;
  weight_ = new float[weight_n_];
  weight_d_ = new float[weight_n_];
  for (int i=0; i<weight_n_; i++) {
    weight_[i] = Randomfunction();
    weight_d_[i] = 0;
  }
  bias_ = Randomfunction();
  bias_d_ = 0;
}

Kernel & Kernel::operator=(const Kernel &assign) {
  if (this == &assign)
    return *this;
  delete [] weight_;
  delete [] weight_d_;

  channel_ = assign.channel_;
  kernel_size_ = assign.kernel_size_;
  weight_n_ = channel_ * kernel_size_ * kernel_size_;
  weight_ = new float[weight_n_];
  weight_d_ = new float[weight_n_];
  for (int i=0; i<weight_n_; i++) {
    weight_[i] = assign.weight_[i];
    weight_d_[i] = assign.weight_d_[i];
  }
  bias_ = assign.bias_;
  bias_d_ = assign.bias_d_;
  return *this;
}

Kernel::~Kernel() {
  delete [] weight_;
  delete [] weight_d_;
}
void Kernel::ZeroGrad() {
  for (int i=0; i<weight_n_; i++) {
    weight_d_[i] = 0;
  }
  bias_d_ = 0;
}

// @ class Linear
Linear::Linear(int batch_size, int input_n, int output_n) {
  evaluator_ = nullptr;
  batch_size_ = batch_size;
  input_n_ = input_n;
  output_n_ = output_n;
  weights_ = new Weight[batch_size_];
  Weight seed(input_n_, output_n_);
  for (int i=0; i<batch_size_; i++) {
    weights_[i] = seed;   // all batch use same weight
  }
}

// Linear::Linear(seal::Evaluator* evaluator, FloatTensor &input, FloatTensor &output) :
//   input_(input), output_(output), evaluator_(evaluator)
// {
//   evaluator_ = nullptr;

//   input_n_ = input.size_[1];
//   output_n_ = output.size_[1];
//   batch_size_ = input.size_[0];
//   weights_ = new Weight[batch_size_];
//   for (int i=0; i<batch_size_; i++) {
//     // Weight t(input_.size_[1], output_.size_[1]);
//     // weights_[i] = t;
//     weights_[i] = *new Weight(input_n_, output_n_);   /// ???????????????????????
//   }

// }

Linear::~Linear() {
  delete [] weights_;
}

void Linear::ZeroGrad(FloatTensor &input) {
  input.ZeroGrad();
  for (int i=0; i<batch_size_; i++) {
    weights_[i].ZeroGrad();
  }
}

// use seal::Evaluator
void Linear::Forward(FloatTensor &input, FloatTensor &output) {
  for (int batch=1; batch<=batch_size_; batch++) {
    for (int i=1; i<=output_n_; i++) {
      float sum = 0;
      for (int j=1; j<=input_n_; j++) {
        sum += input.Get({batch, j}) * weights_[batch].Get(i, j);
      }
      sum += weights_[batch].bias_[i];
      output.Set({batch, i}, sum);
    }
  }
}
// use seal::Evaluator
void Linear::Backward(FloatTensor &input, FloatTensor &output) {
  for (int batch=1; batch<=batch_size_; batch++) {
    // update input tensor grad
    for (int i=1; i<=input_n_; i++) {
      float temp = 0;
      for (int j=1; j<=output_n_; j++) {
        temp += output.GetGrad({batch, j}) * weights_[batch].Get(j, i);
      }
      input.SetGrad({batch, i}, temp);
    }
    // update weight grad
    for (int i=1; i<output_n_; i++) {
      float grad = output.GetGrad({batch, i});
      for (int j=1; j<=input_n_; j++) {
        // 
        weights_[batch].AddGrad(i, j, input.Get({batch, j}) * grad);
      }
      // update bias grad
      weights_[batch].bias_d_[i] += output.GetGrad({batch, i});
    }
  }
}



// @ class Conv2D
// Tensor size is {batch_size, channel_size, height, weight}
Conv2D::Conv2D(int batch_size, int input_channel, int output_channel, int kernel_size, int stride) {
  batch_size_ = batch_size;
  input_channel_ = input_channel;
  output_channel_ = output_channel;
  kernel_size_ = kernel_size;
  stride_ = stride;
  kernel_n_ = batch_size * output_channel;
  kernels_ = new Kernel[kernel_n_];
  for (int channel=0; channel<output_channel_; channel++) {
    Kernel seed(input_channel_, kernel_size);
    for (int batch=0; batch<batch_size_; batch++) {
      int idx = batch * batch_size_ + channel;
      kernels_[idx] = seed;
    }
  }
}

// Conv2D::Conv2D(FloatTensor &input, FloatTensor &output, int kernel_size, int stride) :
//   input_(input), output_(output), kernel_size_(kernel_size), stride_(stride) 
// {
//   // input.batch_size = output.batch_size
//   batch_size_ = input.size_[0];
//   input_channel_ = input.size_[1];
//   input_h_ = input.size_[2];
//   input_w_ = input.size_[3];
//   output_channel_ = output.size_[1];
//   output_h_ = output.size_[2];
//   output_w_ = output.size_[3];
//   kernel_n_ = batch_size_ * output_channel_;
//   kernels_ = new Kernel[kernel_n_];
//   for (int channel=0; channel<output_channel_; channel++) {
//     Kernel seed(input_channel_, kernel_size);
//     for (int batch=0; batch<batch_size_; batch++) {
//       int idx = channel + batch * batch_size_;
//       kernels_[idx] = seed;
//     }
//   }
// }
Conv2D::~Conv2D() {
  delete [] kernels_;
}

float Conv2D::convolution(const Kernel &kernel, const float* const* const data) {
  int idx = 0;
  float sum = 0;
  for (int c=0; c<kernel.channel_; c++) {
    for (int h=0; h<kernel.kernel_size_; h++) {
      for (int w=0; w<kernel.kernel_size_; w++) {
        idx = c * kernel.channel_* + h * kernel.kernel_size_ + w;
        sum += kernel.weight_[idx] * *data[idx];
      }
    }
  }
  sum += kernel.bias_;
  return sum;
}

void Conv2D::expandmap(FloatTensor &input, FloatTensor &output, int stride) {
  // stride
  // define h == w
  // input: x { 3 * 3 }
  // output: stride = 2, kernel = 3, pad = kernel - 1
  // 0 0 0 0 0 0 0 0 0 
  // 0 0 0 0 0 0 0 0 0
  // 0 0 x 0 x 0 x 0 0
  // 0 0 0 0 0 0 0 0 0
  // 0 0 x 0 x 0 x 0 0
  // 0 0 0 0 0 0 0 0 0
  // 0 0 x 0 x 0 x 0 0
  // 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0

  int input_h = input.size_[input.dim_ - 2];
  int input_w = input.size_[input.dim_ - 1];
  int pad_num = input.data_size_ / input_h / input_w;
  int output_h = output.size_[output.dim_ - 2];
  int output_w = output.size_[output.dim_ - 2];
  int pad = (output_h - input_h - (input_h-1)*(stride-1)) / 2;
  // map { h , w }
  int input_idx = 0;
  int output_idx = 0;
  for (int map=0; map<pad_num; map++) {
    for (int p=0; p<pad; p++) {
      for (int i=0; i<output_w; i++) {
        // output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
    }
    for (int h=0; h<input_h; h++) {
      for (int p=0; p<pad; p++) {
        // output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
      for (int w=0; w<input_w; w++) {
        // output.data_[output_idx] = input.data_[input_idx];
        output.grad_[output_idx++] = input.grad_[input_idx++];
        // last w do not add stride
        if (w < input_w) {
          for (int s=0; s<stride-1; s++) {
            output.grad_[output_idx++] = 0;
          }
        }
      }
      for (int p=0; p<pad; p++) {
        // output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
      // last h do not add stride
      if (h < input_h-1) {
        for (int s=0; s<stride-1; s++) {
          for (int w=0; w<output_w; w++) {
            output.grad_[output_idx++] = 0;
          }
        }
      }
    }
    for (int p=0; p<pad; p++) {
      for (int i=0; i<output_w; i++) {
        // output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
    }
  }
  // here should input_idx = input.data_size_;
  //             outpupt_idx = output.data_size_; 
  if (input_idx != input.data_size_ || output_idx != output_idx) {
    std::cout << "ZeroPad2D error" << std::endl;
  }
}

void Conv2D::transkernels(const Kernel* const k_input, Kernel* k_output, int k_input_n) {
  // no need to deal bias, because bp don not use bias
  //
  //
  int k_output_channel = k_input_n;
  int k_output_n = k_input[0].channel_;
  int k_size = k_input[0].kernel_size_;
  Kernel seed(k_output_channel, k_size);
  int o_idx = 0;
  int i_idx = 0;
  for (int i=0; i<k_output_n; i++) {
    // need flip the kernel
    i_idx = (i+1) * kernel_size_ * kernel_n_ - 1;
    for (int j=0; j<k_output_channel; j++) {
      for (int w=0; w<kernel_size_ * kernel_size_; w++) {
        o_idx = j * kernel_size_ * kernel_size_ + w;
        seed.weight_[o_idx] = k_input[j].weight_[i_idx-w];
        seed.weight_d_[o_idx] = k_input[j].weight_d_[i_idx-w];
        o_idx++;
        i_idx++;
      }
    }
    k_output[i] = seed;
  }
}

void Conv2D::tensor2kernels(FloatTensor &input, Kernel* kernels, int stride) {
  // input.h == input.w
  int kernel_n =  input.size_[0];
  int k_channel = input.size_[1];
  int input_size = input.size_[2];
  int k_size = input_size  + (input_size-1)*(stride-1);
  Kernel seed(k_channel, k_size);
  for (int n=0; n<kernel_n; n++) {
    for (int ch=0; ch<k_channel; ch++) {
      int idx = 0;
      for (int h=0; h<input_size; h++) {
        for (int w=0; w<input_size; w++) {
          seed.weight_[idx++] = input.GetGrad({n+1, ch+1,h+1, w+1});    // ???????????????????????????
          if (w < input_size-1) {
            for (int s=0; s<stride; s++) {
              seed.weight_[idx++] = 0;
            }
          }
        }
        if (h < input_size-1) {
          for (int s=0; s<stride; s++) {
            seed.weight_[idx++] = 0;
          }
        }
      }
    }
    kernels[n] = seed;
  }
}

void Conv2D::ZeroGrad(FloatTensor &input) {
  input.ZeroGrad();
  for (int i=0; i<kernel_n_; i++) {
    kernels_[i].ZeroGrad();
  }
}

void Conv2D::Forward(FloatTensor &input, FloatTensor &output) {
  //       w
  //    *******
  //  h *******
  //    *******
  // Todo: use tensor.selection()
  int input_h = input.size_[2];
  int input_w = input.size_[3];

  for (int batch=1; batch<=batch_size_; batch++) {
    int k = 0;
    int idx = 0;
    float* kernel_map[kernels_[0].weight_n_] {nullptr};  // { channel, kernel_h, kernel_w }
    int out_h = 0;
    for (int h=0; h < input_h - kernel_size_ + 1; h += stride_) {
      int out_w = 0;
      for (int w=0; w < input_w - kernel_size_ + 1; w += stride_) {
        for (int in_ch=0; in_ch<input_channel_; in_ch++) {
          for (int i=0; i<kernel_size_; i++) {
            for (int j=0; j<kernel_size_; j++) {
              idx = (batch-1) * (input_channel_*input_h*input_w) +  in_ch * (input_h*input_w) + (h+i) *input_h + w+j;
              kernel_map[k++] = &input.data_[idx];
            }
          }
        }
        for (int out_ch=0; out_ch<output_channel_; out_ch++) {
          output.Set({batch, out_ch+1, out_h, out_w}, convolution(kernels_[out_ch], kernel_map));
        }
        k = 0;
      }    
    }
  }
}

void Conv2D::Backward(FloatTensor &input, FloatTensor &output) {
  // out_exp_h = out_exp_w
  int input_h = input.size_[2];
  int input_w = input.size_[3];
  int output_h = output.size_[2];
  int output_w = output.size_[3];

  int out_exp_h = input_h + kernel_size_ - 1;
  FloatTensor out_exp({batch_size_, output_channel_, out_exp_h, out_exp_h});
  expandmap(output, out_exp, stride_);

  Kernel kernels_trans[input_channel_];
  transkernels(kernels_, kernels_trans,  kernel_n_);

  for (int batch=1; batch<=batch_size_; batch++) {
    // update input tensor grad
    int k = 0;
    int idx = 0;
    float* kernel_map[kernels_[0].weight_n_] {nullptr};  // { channel, kernel_h, kernel_w }
    int in_h = 0;
    for (int h=0; h < out_exp_h - kernel_size_ + 1; h++) {
      int in_w = 0;
      for (int w=0; w < out_exp_h - kernel_size_ + 1; w++) {
        // output_channel == output_exp_channel
        for (int out_ch=0; out_ch<output_channel_; out_ch++) {
          for (int i=0; i<kernel_size_; i++) {
            for (int j=0; j<kernel_size_; j++) {
              idx = (batch-1) * (output_channel_*out_exp_h*out_exp_h) +  out_ch * (out_exp_h*out_exp_h) + (h+i) *out_exp_h + w+j;
              kernel_map[k] = &out_exp.grad_[idx];
            }
          }
        }
        for (int in_ch=0; in_ch<input_channel_; in_ch++) {
          // grad - kernel.bias  ???????????????????
          input.SetGrad({batch, in_ch+1, in_h, in_w}, convolution(kernels_trans[in_ch], kernel_map)-kernels_trans[in_ch].bias_);
        }
        k = 0;
      }
    }  
  }

  // update kernel grad

  Kernel kernels_exp[batch_size_];
  tensor2kernels(output, kernels_exp, stride_);
  int kernel_exp_size = kernels_exp[0].kernel_size_;

  for (int batch=1; batch<=batch_size_; batch++) {
    for (int k=0; k<kernel_n_; k++) {
      int k_h = 0;
      for (int h=0; h < input_h - kernel_exp_size + 1; h++) {
        int k_w = 0;
        for (int w=0; w < input_w - kernel_exp_size + 1; w++) {
          float grad = 0;
          for (int k_exp_ch=0; k_exp_ch < kernel_exp_size; k_exp_ch++) {
            for (int k_exp_h=0; k_exp_h < kernel_exp_size; k_exp_h++) {
              for (int k_exp_w=0; k_exp_w < kernel_exp_size; k_exp_w++) {
                grad += input.Get({batch, k+1, h+k_exp_h+1, w+k_exp_w+1}) * kernels_exp[batch].weight_d_[k_exp_h * kernel_exp_size + k_exp_w];
              }
            }
          }
          kernels_[batch].weight_d_[h * kernel_size_ + w] = grad;
          grad = 0;
        }
      }
    }
  }

  // update kernel bias
  for (int batch=1; batch<=batch_size_; batch++) {
    for (int ch=1; ch<=output_channel_; ch++) {
      float grad = 0;
      for (int h=1; h<=output_h; h++) {
        for (int w=1; w<=output_w; w++) {
          grad += output.GetGrad({batch, ch, h, w});
        }
      }
      kernels_[batch].bias_d_ = grad;
    }
  }
}

// @ class AvgPool2D
// Tensor size is {batch_size, channel_size, height, weight}
AvgPool2D::AvgPool2D(int kernel_size, int stride) {
  kernel_size_ = kernel_size;
  stride_ = stride;
}

// AvgPool2D::AvgPool2D(FloatTensor &input, FloatTensor &output, int kernel_size) :
//   input_(input), output_(output), kernel_size_(kernel_size), stride_(kernel_size)
// {
//   // input.batch_size = output.batch_size
//   // input.channel = output.channel
//   batch_size_ = input.size_[0];
//   channel_ = input.size_[1];
//   input_w_ = input.size_[2];
//   input_h_ = input.size_[3];
// }

void AvgPool2D::ZeroGrad(FloatTensor &input) {
  input.ZeroGrad();
}

float AvgPool2D::pool(float* kernel, int n) {
  float temp = 0;
  for (int i=0; i<n; i++) {
    temp += kernel[i];
  }
  return temp / n;
}

void AvgPool2D::Forward(FloatTensor &input, FloatTensor &output) {
  //       w
  //    *******
  //  h *******
  //    *******
  int batch_size = input.size_[0];
  int channel = input.size_[1];
  int input_h = input.size_[2];
  int input_w = input.size_[3];

  float kernel[kernel_size_ * kernel_size_] {0};
  for (int batch=1; batch<=batch_size; batch++) {
    for (int ch=1; ch<=channel; ch++) {
      int output_h = 1;
      for (int h=0; h < input_h - stride_; h += stride_) {
        int output_w = 1;
        for (int w=0; w < input_w - stride_; w += stride_) {
          for (int k=0; k < kernel_size_ * kernel_size_; k++) {
            for (int i=0; i<stride_; i++) {
              for (int j=0; j<stride_; j++) {
                kernel[k] = input.Get({batch, ch, h+i+1, w+j+1});
              }
            }
          }
          output.Set({batch, ch, output_h, output_w}, pool(kernel, kernel_size_ * kernel_size_));
          output_w++;
        }
        output_h++;
      }
    }
  }
}

void AvgPool2D::Backward(FloatTensor &input, FloatTensor &output) {
  // only propagation error to pre layer, average pooling equal division error to kernel    ????
  int batch_size = output.size_[0];
  int channel = output.size_[1];
  int output_h = output.size_[2];
  int output_w = output.size_[3];


  for (int batch=1; batch<=batch_size; batch++) {
    for (int ch=1; ch<=channel; channel++) {
      int input_h = 1;
      for (int h=0; h < output_h; h++) {
        int input_w = 1;
        for (int w=0; w < output_w; w++) {
          float grad = output.GetGrad({batch, ch, h+1, w+1});
          grad /= kernel_size_ * kernel_size_;
         
          for (int i=0; i<stride_; i++) {
            for (int j=0; j<stride_; j++) {
                input.SetGrad({batch, ch, input_h, input_w}, grad);
            }
          }
          input_w++;
        }
        input_h++;
      }
    }
  }

}

// @ class Flatten
Flatten::Flatten() {

}

void Flatten::ZeroGrad(FloatTensor &input) {
  input.ZeroGrad();
}

void Flatten::Forward(FloatTensor &input, FloatTensor &output) {
  for (int i=0; i<input.data_size_; i++) {
    output.data_[i] = input.data_[i];
  }
}

void Flatten::Backward(FloatTensor &input, FloatTensor &output) {
  for (int i=0; i<output.data_size_; i++) {
    input.grad_[i] = output.grad_[i];
  }
}

// @ class Relu
// Tensor size is {batch_size, channel_size, height, weight}
Relu::Relu() {

}

// Relu::Relu(FloatTensor &input, FloatTensor &output) :
//   input_(input), output_(output)
// {
//   batch_size_ = input_.size_[0];
// }

inline float Relu::relu(float x) {
  return (x > 0) ? x : 0;
}

inline float Relu::relu_d(float x) {
  return (x > 0) ? 1 : 0;
}

void Relu::ZeroGrad(FloatTensor &input) {
  input.ZeroGrad();
}

void Relu::Forward(FloatTensor &input, FloatTensor &output) {
  for (int i=0; i<input.data_size_; i++) {
    output.data_[i] = relu(input.data_[i]);
  }
}

void Relu::Backward(FloatTensor &input, FloatTensor &output) {
  for (int i=0; i<input.data_size_; i++) {
    input.grad_[i] = output.grad_[i] * relu_d(input.data_[i]);
  }
}

// @ class Sigmoid
// Tensor size is {batch_size, channel_size, height, weight}
Sigmoid::Sigmoid(FloatTensor &input, FloatTensor &output) :
  input_(input), output_(output)
{
  batch_size_ = input_.size_[0];
}

inline float Sigmoid::sigmoid(float x) {
  return 1 / (1 + std::exp(-x));
}

inline float Sigmoid::sigmoid_d(float x) {
  float temp = sigmoid(x);
  return temp * (1 - temp);
}

void Sigmoid::Forward() {
  // for (int batch=0; batch<batch_size_; batch++) {
  // }
  for (int i=0; i<input_.data_size_; i++) {
    output_.data_[i] = sigmoid(input_.data_[i]);
  }
}

void Sigmoid::Backward() {
  for (int i=0; i<input_.data_size_; i++) {
    input_.grad_[i] = output_.grad_[i] * sigmoid_d(input_.data_[i]);
  }
}

// @ class Softmax
// Tensor size is {batch_size, class_num}
Softmax::Softmax () {

}

// Softmax::Softmax(FloatTensor &input, FloatTensor &output) :
//   input_(input), output_(output)
// {  // input ==  output
//   batch_size_ = input.size_[0];
//   class_num_ = input.size_[1];
// }

inline void Softmax::softmax(float* x, int n) {
  float sum = 0;
  for (int i=0; i<n; i++) {
    x[i] = std::exp(x[i]);
    sum += x[i];
  }
  for (int i=0; i<n; i++) {
    x[i] /= sum;
  }
}

inline void Softmax::softmax_d(float* x, int n) {
  softmax(x, n);
  for (int i=0; i<n; i++) {
    x[i] = x[i] * (1 - x[i]);
  }
}
// Todo: exp(xi - max(x))
void Softmax::Forward(FloatTensor &input, FloatTensor &output) {
  int batch_size = input.size_[0];
  int class_num = input.size_[1];

  float temp[class_num] = {0};
  for (int batch=1; batch<batch_size; batch++) {
    for (int i=0; i<class_num; i++) {
      temp[i] = input.Get({batch, i+1});
    }
    softmax(temp, class_num);
    for (int i=0; i<class_num; i++) {
      output.Set({batch, i+1}, temp[i]);
    }
  }
}
// Todo 
void Softmax::Backward(FloatTensor &input, FloatTensor &output) {
  int batch_size = output.size_[0];
  int class_num = output.size_[1];

  float temp[class_num] = {0};
  for (int batch=1; batch<batch_size; batch++) {
    for (int i=0; i<class_num; i++) {
      temp[i] = input.GetGrad({batch, i+1});
    }
    softmax_d(temp, class_num);
    for (int i=0; i<class_num; i++) {
      output.SetGrad({batch, i+1}, temp[i]);
    }
  }

}

// @ class LogSoftmax
// Tensor size is {batch_size, class_num}
LogSoftmax::LogSoftmax () {

}

inline void LogSoftmax::log_softmax(float* x, int n) {
  float sum = 0;
  for (int i=0; i<n; i++) {
    x[i] = std::exp(x[i]);
    sum += x[i];
  }
  for (int i=0; i<n; i++) {
    x[i] /= sum;
  }
}



NLLLoss::NLLLoss() {

}

// @ class CrossEntropyLoss
CrossEntropyLoss::CrossEntropyLoss() {

}



float CrossEntropyLoss::Loss(FloatTensor &input, FloatTensor &target) {
  int batch_size = input.size_[0];
  int class_num = input.size_[1];
  FloatTensor output({batch_size, class_num});
  Softmax softmax;
  softmax.Forward(input, output);

  float loss = 0;
  for (int batch=1; batch<=batch_size; batch++) {
    int cls = target.Get({batch}) + 1;
    loss += -log(input.Get({batch, cls}));
  }
  Backward(input, output);
  return loss;
}

void CrossEntropyLoss::ZeroGrad(FloatTensor &input) {
  input.ZeroGrad();
}

void CrossEntropyLoss::Forward() {

}

void CrossEntropyLoss::Backward(FloatTensor &input, FloatTensor &output) {
  for (int batch=1; batch<=input.size_[0]; batch++) {
    for (int cls=1; cls<=input.size_[1]; cls++) {
      float grad = output.Get({batch, cls}) - input.Get({batch, cls});
      input.SetGrad({batch, cls}, grad);
    }
  }
}

// @ function Pad()
void ZeroPading2D(const FloatTensor &input, FloatTensor&output, int stride) {
  // input tensor { a, b, ... , h, w }
  // output tensor { a, b, ..., h + 2*pad, w + 2*pad }
  // define h == w
  int input_h = input.size_[input.dim_ - 2];
  int input_w = input.size_[input.dim_ - 1];
  int pad_num = input.data_size_ / input_h / input_w;
  int output_h = output.size_[output.dim_ - 2];
  int output_w = output.size_[output.dim_ - 2];
  int pad = (output_h - input_h) / 2;
  // map { h , w }
  int input_idx = 0;
  int output_idx = 0;
  for (int map=0; map<pad_num; map++) {
    for (int p=0; p<pad; p++) {
      for (int i=0; i<output_w; i++) {
        output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
    }
    for (int h=0; h<input_h; h++) {
      for (int p=0; p<pad; p++) {
        output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
      for (int w=0; w<input_w; w++) {
        output.data_[output_idx] = input.data_[input_idx];
        output.grad_[output_idx++] = input.grad_[input_idx++];
      }
      for (int p=0; p<pad; p++) {
        output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
    }
    for (int p=0; p<pad; p++) {
      for (int i=0; i<output_w; i++) {
        output.data_[output_idx] = 0;
        output.grad_[output_idx++] = 0;
      }
    }
  }
  // here should input_idx = input.data_size_;
  //             outpupt_idx = output.data_size_; 
  if (input_idx != input.data_size_ || output_idx != output_idx) {
    std::cout << "ZeroPad2D error" << std::endl;
  }
}

}