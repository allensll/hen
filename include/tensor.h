#pragma once

#include <seal/seal.h>
#include <initializer_list>

namespace hen {
// // @ class Tensor
// class Tensor {
//  public:
  
// }
// @ class IntTensor
class IntTensor {
 public:
  int dim_;
  int* size_;
  int* data_;
  int data_size_;
  IntTensor(std::initializer_list<int> il);
  IntTensor(int dim, int size[]);
  IntTensor(int dim, int size[], int data[]);
  ~IntTensor();
};
class FloatTensor {
 public:
  int dim_;
  int* size_;
  int* idx_size_;
  float* data_;
  int data_size_;

  float* grad_;

  FloatTensor();
  FloatTensor(std::initializer_list<int> il);
  FloatTensor & operator=(const FloatTensor &assign);
  ~FloatTensor();
  const float Get(std::initializer_list<int> il);
  const float GetGrad(std::initializer_list<int> il);
  void Set(std::initializer_list<int> il, float x);
  void SetGrad(std::initializer_list<int> il, float x);
  void ZeroGrad();
  // Todo: return address
  void Selection(std::initializer_list<int> il, float* data[], int n);
  // Todo: return address
  void SelectionGrad(std::initializer_list<int> il, float* data[], int n);
  // void View(std::initializer_list<int> il);
};

// @ class CipherTensor
class CipherTensor {
 public:
  int dim_;
  int* size_;
  seal::Ciphertext* data_;
  int data_size_;
  CipherTensor(std::initializer_list<int> il);
  CipherTensor(int dim, int size[]);
  CipherTensor(int dim, int size[], seal::Ciphertext data[]);
  ~CipherTensor();

};

}