#include <iostream>
#include "hen.h"
#include "optim.h"

using namespace std;

int main()
{
  /* code */
  // for (int i=0; i<10; i++) {
  //   cout << hen::Randomfunction() << " ";
  // }
  // cout << endl;

  hen::MNIST dataset("mnist");
  dataset.LoadMNIST();
  int batch_size = 2;
  hen::FloatTensor input_batches[60000/batch_size] {};  // { batch_size, 1, height, weight }
  hen::FloatTensor label_batches[60000/batch_size] {};  // { batch_size }
  dataset.GetTrainBatches(input_batches, label_batches, 60000/batch_size, batch_size);

  // hen::FloatTensor input({2, 1, 28, 28});
  hen::FloatTensor output({batch_size, 10});

  hen::CNN model(2);
  model.Forward(input_batches[0], output);

  for (int b=1; b<=batch_size; b++) {
    for (int i=1; i<=10; i++) {
      cout << output.Get({1, i}) << " ";
    }
    cout << endl;
  }
  
  hen::CrossEntropyLoss cel;
  float loss = cel.Loss(output, label_batches[0]);
  cout << "loss: " << loss << endl;
  model.Backward(input_batches[0], output);









  // hen::MNIST dataset("mnist");
  // dataset.LoadMNIST();
  // int batch_size = 2;
  // hen::FloatTensor input_batches[60000/batch_size] {};  // { batch_size, 1, height, weight }
  // hen::FloatTensor label_batches[60000/batch_size] {};  // { batch_size }
  // dataset.GetTrainBatches(input_batches, label_batches, 60000/batch_size, batch_size);

  // hen::FloatTensor output({batch_size, 10});

  // hen::CNN model(2);
  // hen::CrossEntropyLoss cel;
  // hen::SGD optimizer(0.01, 0.9);

  // float loss = 0;
  // int epoch = 3;
  // for (int e=0; e<epoch; e++) {
  //   // train
  //   for (int batch; batch<60000/batch_size; batch++) {
  //     model.Forward(input_batches[batch], output);
  //     loss = cel.Loss(output, label_batches[batch]);
  //     model.Backward(input_batches[batch], output);
  //     optimizer.Step(model);
  //   }
  //   // predict
  // }






  return 0;
}
