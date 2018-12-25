#include <fstream>
#include <filesystem>
#include "dataset.h"

using namespace std;
typedef filesystem::path Path;

namespace hen {
// @class MNIST
int MNIST::TranslateEndian_32(char* const buffer) {
  int res = (uint8_t)buffer[0] << 24 | (uint8_t)buffer[1] << 16 | (uint8_t)buffer[2] << 8 | (uint8_t)buffer[3];
  return res;
}

MNIST::MNIST(string dataset_path) {

  height_ = 28;
  weight_ = 28;
  train_numbers_ = 60000;
  test_numbers_ = 10000;

  dataset_path_ = dataset_path;
  train_data_name_ = "train-images-idx3-ubyte";
  train_label_name_ = "train-labels-idx1-ubyte";
  test_data_name_ = "t10k-images-idx3-ubyte";
  test_label_name_ = "t10k-labels-idx1-ubyte";

  train_ = new Sample[train_numbers_];
  for (int i=0; i<train_numbers_; i++) {
    train_[i].data = new int[height_*weight_] {};
  }
  test_ = new Sample[test_numbers_];
  for (int i=0; i<test_numbers_; i++) {
    test_[i].data = new int[height_*weight_] {};
  }
}

MNIST::~MNIST() {
  for (int i=0; i<train_numbers_; i++) {
    delete [] train_[i].data;
  }
  delete [] train_;
  for (int i=0; i<test_numbers_; i++) {
    delete [] test_[i].data;
  }
  delete [] test_;
}

void MNIST::LoadData(Sample* samples, const string file_name) {
  ifstream data_file(file_name, ios_base::in | ios_base::binary);
  if (!data_file.is_open()) {
    cout << "Unable to open : " << file_name << endl;
    return;
  }
  char* buffer = new char[sizeof(int)];
  int magic_number = 0;
  int sample_number = 0;
  int n_rows = 0, n_cols = 0;
  data_file.read(buffer, sizeof(int));
  magic_number = TranslateEndian_32(buffer);
  data_file.read(buffer, sizeof(int));
  sample_number = TranslateEndian_32(buffer);
  data_file.read(buffer, sizeof(int));
  n_rows = TranslateEndian_32(buffer);
  data_file.read(buffer, sizeof(int));
  n_cols = TranslateEndian_32(buffer);
    
  cout << "----------------------" << endl;
  cout << "Sample number : " << sample_number << endl;
  cout << "Height : " << n_rows << " pixels" << endl;
  cout << "Weight : " << n_cols << " pixels" << endl;
  cout << "----------------------" << endl;

  // int zero_padding = 2;
  // int padded_matrix_size = (n_cols + 2 * zero_padding) * (n_rows + 2 * zero_padding);
  unsigned char temp;
  // double normalize_max = 1, normalize_min = -1;

  // double normalize_max = 1.175, normalize_min = -0.1;

  // test
  // sample_number = 1;

  for(int k = 0; k < sample_number; k++){
    // samples[k].data = (double *)malloc(padded_matrix_size * sizeof(double));
    // memset(samples[k].data, 0, padded_matrix_size * sizeof(double));
    for(int i = 0; i < n_rows; i++){
      for(int j = 0; j < n_cols; j++){
        data_file.read((char*)&temp, 1);
        // cout << i << "  "<< j << "---" << (double)temp/255 << endl;
        // samples[k].data[(i + zero_padding) * n_cols + j + zero_padding] = (int)temp/255 * (normalize_max - normalize_min) + normalize_min;//把灰度归一化到[0, 1]
        samples[k].data[i * n_cols + j] = (int)temp;
      }
    }
  }
  data_file.close();
}

void MNIST::LoadLabel(Sample* samples, const string file_name) {
  ifstream data_file(file_name, ios_base::in | ios_base::binary);
  if (!data_file.is_open()) {
    cout << "Unable to open : " << file_name << endl;
    return;
  }

  char* buffer = new char[sizeof(int)];
  int magic_number = 0;
  int sample_number = 0;
  data_file.read((char*)buffer, sizeof(int));
  magic_number = TranslateEndian_32(buffer);
  data_file.read((char*)buffer, sizeof(int));
  sample_number = TranslateEndian_32(buffer);

  unsigned char temp;

  for(int k = 0; k < sample_number; k++){
    data_file.read((char*)&temp, 1);
    samples[k].label = (int)temp;
  }
  data_file.close();
}

void MNIST::LoadMNIST() {
  Path dir(dataset_path_);

  LoadData(train_, dir / Path(train_data_name_));
  LoadLabel(train_, dir / Path(train_label_name_));

  LoadData(test_, dir / Path(test_data_name_));
  LoadLabel(test_, dir / Path(test_label_name_));
}

void MNIST::GetTrainSamples(IntTensor &data, IntTensor &label) {
  // data { 60000, 28, 28 }
  // label { 60000 }
  int wh = weight_ * height_;
  for (int i=0; i<train_numbers_; i++) {
    for (int j=0; j<wh; j++) {
      data.data_[i*wh+j] = train_[i].data[j];
    }
    label.data_[i] = train_[i].label;
  }
}

void MNIST::GetTestSamples(IntTensor &data, IntTensor &label) {
  // data { 10000, 28, 28 }
  // label { 10000 }
  int wh = weight_ * height_;
  for (int i=0; i<test_numbers_; i++) {
    for (int j=0; j<wh; j++) {
      data.data_[i*wh+j] = test_[i].data[j];
    }
    label.data_[i] = test_[i].label;
  }
}

void MNIST::GetTrainBatches(FloatTensor data[], FloatTensor label[], int batch_n, int batch_size) {
  int wh = weight_ * height_;
  for (int batch=0; batch<batch_n; batch++) {
    FloatTensor data_temp;
    FloatTensor label_temp;
    for (int i=0; i<batch_size; i++) {
      for (int j=0; j<wh; j++) {
        data_temp.data_[i*wh+j] = train_[batch*batch_size+i].data[i*wh+j];
      }
      label_temp.data_[i] = train_[batch*batch_size].label;
    }
    data[batch] = data_temp;
    label[batch] = label_temp;
  }
}

void MNIST::GetTestBatches(FloatTensor data[], FloatTensor label[], int batch_n, int batch_size) {
  int wh = weight_ * height_;
  for (int batch=0; batch<batch_n; batch++) {
    FloatTensor data_temp;
    FloatTensor label_temp;
    for (int i=0; i<batch_size; i++) {
      for (int j=0; j<wh; j++) {
        data_temp.data_[i*wh+j] = test_[batch*batch_size+i].data[i*wh+j];
      }
      label_temp.data_[i] = test_[batch*batch_size].label;
    }
    data[batch] = data_temp;
    label[batch] = label_temp;
  }
}

}