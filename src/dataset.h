#pragma once

#include <iostream>
#include "tensor.h"

using namespace std;

namespace hen {
// @struct Sample
struct Sample {
  int* data; // height * weight
  int label;
};

// @class MNIST
class MNIST {
 public:
  int height_;
  int weight_;
  int class_num_;
  int train_numbers_;
  int test_numbers_;

  string dataset_path_;

  string train_data_name_;
  string train_label_name_;
  string test_data_name_;
  string test_label_name_;

  static int TranslateEndian_32(char* const buffer);

  MNIST(string dataset_path);
  ~MNIST();
  void LoadMNIST();
  void GetTrainSamples(IntTensor &data, IntTensor &label);  // void GetTrainSamples( Tensor )
  void GetTestSamples(IntTensor &data, IntTensor &label);
  void GetTrainBatches(FloatTensor data[], FloatTensor label[], int batch_n, int batch_size);  // batch_n*batch_size=train_number
  void GetTestBatches(FloatTensor data[], FloatTensor label[], int batch_n, int batch_size);   // batch_n*batch_size=test_number

 private:
  Sample* train_;
  Sample* test_;
  
  void LoadData(Sample* samples, const string file_name);
  void LoadLabel(Sample* samples, const string file_name);
  // Sample* GetSample(int idx);

};

}