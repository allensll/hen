#include "models.h"

namespace hen {
// @ class CNN
// input tensor { batch_size, 1, 28, 28 }
// output tensor { batch, 10 }
CNN::CNN(int batch_size) :
  output_conv1_({batch_size, 20, 24, 24}),
  output_relu1_({batch_size, 20, 24, 24}),
  output_pool1_({batch_size, 20, 12, 12}),
  output_conv2_({batch_size, 50, 8, 8}),
  output_relu2_({batch_size, 50, 8, 8}),
  output_pool2_({batch_size, 50, 4, 4}),
  output_flat1_({batch_size, 50*4*4}),
  output_fc1_({batch_size, 500}),
  output_relu3_({batch_size, 500}),
  // output_fc2_({batch_size, 10}),

  conv1_(Conv2D(batch_size, 1, 20, 5, 1)),
  relu1_(Relu()),
  pool1_(AvgPool2D(2, 2)),
  conv2_(Conv2D(batch_size, 20, 50, 5, 1)),
  relu2_(Relu()),
  pool2_(AvgPool2D(2, 2)),
  flat1_(Flatten()),
  fc1_(Linear(batch_size, 50*4*4, 500)),
  relu3_(Relu()),
  fc2_(Linear(batch_size, 500, 10))
  // softmax1_(Softmax())
{
  batch_size_ = batch_size;
}

// Run or Forward   ??????
void CNN::Run(FloatTensor &input, FloatTensor &output) {
  // conv1_()
}

void CNN::Forward(FloatTensor &input, FloatTensor &output) {
  conv1_.Forward(input, output_conv1_);
  relu1_.Forward(output_conv1_, output_relu1_);
  pool1_.Forward(output_relu1_, output_pool1_);
  conv2_.Forward(output_pool1_, output_conv2_);
  relu2_.Forward(output_conv2_, output_relu2_);
  pool2_.Forward(output_relu2_, output_pool2_);
  flat1_.Forward(output_pool2_, output_flat1_);
  fc1_.Forward(output_flat1_, output_fc1_);
  relu3_.Forward(output_fc1_, output_relu3_);
  fc2_.Forward(output_relu3_, output);
  // softmax1_.Forward(output_fc2_, output);
}

void CNN::Backward(FloatTensor &input, FloatTensor &output) {
  // softmax1_.Backward(output_fc2_, output);
  fc2_.Backward(output_relu3_, output);
  relu3_.Backward(output_fc1_, output_relu3_);
  fc1_.Backward(output_flat1_, output_fc1_);
  flat1_.Backward(output_pool2_, output_flat1_);
  pool2_.Backward(output_relu3_, output_pool2_);
  relu2_.Backward(output_conv2_, output_relu2_);
  conv2_.Backward(output_pool1_, output_conv2_);
  pool1_.Backward(output_relu1_, output_pool1_);
  relu1_.Backward(output_conv1_, output_relu1_);
  conv1_.Backward(input, output_relu1_);
}


}