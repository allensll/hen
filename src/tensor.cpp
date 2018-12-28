#include <cstring>
#include "tensor.h"

using namespace std;
// using namespace seal;

namespace hen {
// @ class IntTensor
IntTensor::IntTensor(initializer_list<int> il) {
  dim_ = il.size();
  size_ = new int[dim_];
  data_size_ = 1;
  int i=0;
  for (int n : il) {
    data_size_ *= n;
    size_[i++] = n;
  }
  data_ = new int[data_size_];
}

IntTensor::IntTensor(int dim, int size[]) {
  dim_ = dim;
  size_ = new int[dim];
  data_size_ = 1;
  for (int i=0; i<dim_; i++) {
    data_size_ *= size[i];
    size_[i] = size[i];
  }
  data_ = new int[data_size_];
}

IntTensor::IntTensor(int dim, int size[], int data[]) {
  IntTensor(dim, size, data);
  for (int i=0; i<data_size_; i++) {
    data_[i] = data[i];
  }
}

IntTensor::~IntTensor() {
  delete [] data_;
  delete [] size_;
}

// @ class FloatTensor
FloatTensor::FloatTensor() {
  dim_ = 0;
  size_ = new int[1] {0};
  idx_size_ = new int[1] {0};
  data_ = new float[1] {0};
  data_size_ = 0;
  grad_ = new float[1] {0};
}

FloatTensor::FloatTensor(initializer_list<int> il) {
  dim_ = il.size();
  size_ = new int[dim_];
  data_size_ = 1;
  int i=0;
  for (int n : il) {
    data_size_ *= n;
    size_[i++] = n;
  }
  idx_size_ = new int[dim_+1];
  idx_size_[0] = data_size_;
  i = 1;
  for (int n : il) {
    idx_size_[i++] = idx_size_[i-1] / n;
  }
  data_ = new float[data_size_] {0};
  grad_ = new float[data_size_] {0};
}

FloatTensor & FloatTensor::operator=(const FloatTensor &assign) {
  if (this == &assign)
    return *this;
  delete [] size_;
  delete [] idx_size_;
  delete [] data_;
  delete [] grad_;

  dim_ = assign.dim_;
  data_size_ = assign.data_size_;
  size_ = new int[dim_] {};
  for (int i=0; i<dim_; i++) {
    size_[i] = assign.size_[i];
  }
  idx_size_ = new int[dim_+1];
  for (int i=0; i<dim_+1; i++) {
    idx_size_[i] = assign.idx_size_[i];
  }
  data_ = new float[data_size_];
  for (int i=0; i<data_size_; i++) {
    data_[i] = assign.data_[i];
  }
  grad_ = new float[data_size_];
  for (int i=0; i<data_size_; i++) {
    grad_[i] = assign.grad_[i];
  }
  return *this;
}

FloatTensor::~FloatTensor() {
  // cout << "1" << endl;
  // cout << "size ";
  // for (int i=0; i<dim_; i++) {
  //   cout << " " << size_[i];
  // }
  // cout << endl;
  delete [] size_;
  // cout << "2" << endl;
  delete [] idx_size_;
  // cout << "data " << data_[0] << endl;
  delete [] data_;
  // cout << "4" << endl;
  // cout << "grad " << grad_[0] << endl;
  delete [] grad_;
  // cout << "5" << endl;
}
const float FloatTensor::Get(initializer_list<int> il) {
  if (il.size() > dim_) {
    cout << "Get error, dim mismatch" << endl;
    return 0;
  }
  int i = 0;
  int idx = 0;
  for (int n : il) {
    if (n > size_[i]) {
      cout << "Get error, size mismatch" << endl;
      return 0;
    }
    idx += (n-1) * idx_size_[i+1];
    i++;
  }
  return data_[idx];
}
const float FloatTensor::GetGrad(initializer_list<int> il) {
  if (il.size() > dim_) {
    cout << "GetGrad error, dim mismatch" << endl;
    return 0;
  }
  int i = 0;
  int idx = 0;
  for (int n : il) {
    if (n > size_[i]) {
      cout << "GetGrad error, size mismatch" << endl;
      return 0;
    }
    idx += (n-1) * idx_size_[i+1];
    i++;
  }
  return grad_[idx];
}

void FloatTensor::Set(initializer_list<int> il, float x) {
  if (il.size() > dim_) {
    cout << "Set error, dim mismatch" << endl;
    return;
  }
  int i = 0;
  int idx = 0;
  for (int n : il) {
    if (n > size_[i]) {
      cout << "Set error, size mismatch" << endl;
      return;
    }
    idx += (n-1) * idx_size_[i+1];
    i++;
  }
  data_[idx] = x;
  return;
}
void FloatTensor::SetGrad(initializer_list<int> il, float x) {
  if (il.size() > dim_) {
    cout << "SetGrad error, dim mismatch" << endl;
    return;
  }
  int i = 0;
  int idx = 0;
  for (int n : il) {
    if (n > size_[i]) {
      cout << "SetGrad error, size mismatch" << endl;
      return;
    }
    idx += (n-1) * idx_size_[i+1];
    i++;
  }
  grad_[idx] = x;
}
void FloatTensor::ZeroGrad() {
  for (int i=0; i<data_size_; i++) {
    grad_[i] = 0;
  }
}

void FloatTensor::Selection(initializer_list<int> il, float* data[], int n) {
  // Todo: check the validity of input
  int size = il.size();
  int range[size] {0};
  int idx = 0;
  for (int n : il) {
    range[idx++] = n;
  }
  // int data_size = 1;
  // for (int i=0; i<size; i += 2) {
  //   data_size *= range[i+1] - range[i];
  // }
  for (int i=0; i<n; i++) {
    
  }
}

// FloatTensor FloatTensor::View(initializer_list<int> il) {   ????????????????
//   int data_size = 1;
//   for (int n : il) {
//     data_size *= n;
//   }
//   if (data_size != data_size_) {
//     cout << "view error, size mismatch" << endl;
//     return;
//   }
//   dim_ = il.size();
//   delete [] size_;
//   size_ = new int[dim_];
//   int i=0;
//   for (int n : il) {
//     data_size_ *= n;
//     size_[i++] = n;
//   }

// }



// @ class CipherTensor
CipherTensor::CipherTensor(initializer_list<int> il) {
  dim_ = il.size();
  size_ = new int[dim_];
  data_size_ = 1;
  int i=0;
  for (int n : il) {
    data_size_ *= n;
    size_[i++] = n;
  }
  data_ = new seal::Ciphertext[data_size_];
}

CipherTensor::CipherTensor(int dim, int size[]) {
  dim_ = dim;
  size_ = new int[dim];
  data_size_ = 1;
  for (int i=0; i<dim_; i++) {
    data_size_ *= size[i];
    size_[i] = size[i];
  }
  data_ = new seal::Ciphertext[data_size_];
}

CipherTensor::CipherTensor(int dim, int size[], seal::Ciphertext data[]) {
  CipherTensor(dim, size, data);
  for (int i=0; i<data_size_; i++) {
    data_[i] = data[i];
  }
}

CipherTensor::~CipherTensor() {
  delete [] data_;
  delete [] size_;
}


} // namespace hen