#include <iostream>
#include "hen.h"

using namespace std;

int main()
{
  /* code */
  hen::MNIST dataset("mnist");
  dataset.LoadMNIST();
  int batch_n = 2;
  hen::FloatTensor input_batches[60000/batch_n] {};  // { batch_size, 1, height, weight }
  hen::FloatTensor label_batches[60000/batch_n] {};  // { batch_size }
  dataset.GetTrainBatches(input_batches, label_batches, 60000/2, 2);

  cout << label_batches[21].Get({1}) << endl;
  cout << label_batches[21].Get({2}) << endl;

  // hen::IntTensor input({60000, 28, 28});
  // hen::IntTensor label({60000});
  // dataset.GetTrainSamples(input, label);
  // // dataset.GetTestSamples(input, label);

  // hen::User user = hen::User(2048, 256);
  // hen::CipherTensor cinput({10000, 28, 28});
  // hen::CipherTensor clabel({10000});
  // // user.Encrypt(input, cinput);

  // cout << label.data_[0] << endl;
  // cout << label.data_[10] << endl;
  // cout << label.data_[20] << endl;
  // cout << label.data_[30] << endl;

  // user.Encrypt(label, clabel);

  // hen::IntTensor plabel({10000});
  // user.Decrypt(clabel, plabel);

  // cout << plabel.data_[0] << endl;
  // cout << plabel.data_[10] << endl;
  // cout << plabel.data_[20] << endl;
  // cout << plabel.data_[30] << endl;

  return 0;
}
