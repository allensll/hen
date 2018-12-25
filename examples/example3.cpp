#include <iostream>
#include "hen.h"

using namespace std;

int main()
{
  /* code */
  
  // hen::MNIST dataset("mnist");
  // dataset.LoadMNIST();
  // hen::IntTensor input({10000, 28, 28});
  // hen::IntTensor label({10000});
  // dataset.GetTestSamples(input, label);

  hen::FloatTensor input({2, 1, 28, 28});
  hen::FloatTensor output({2, 10});

  // for (int i=0; i<=5; i++) {
  //   cout << input.idx_size_[i] << endl;
  // }

  for (int i=1; i<=10; i++) {
    output.Set({1, i}, 13);
  }
  for (int i=1; i<=10; i++) {
    cout << output.Get({1, i}) << " ";
  }
  cout << endl;

  hen::CNN model(2);
  model.Forward(input, output);

  for (int i=1; i<=10; i++) {
    cout << output.Get({1, i}) << " ";
  }
  cout << endl;






  int epoch = 3;
  for (int e=0; e<epoch; e++) {
    
  }






  return 0;
}
