#include <iostream>
#include <iomanip>
#include "hen.h"
#include "optim.h"

using namespace std;

int main()
{
  /* code */

  // hen::MNIST dataset("examples/mnist");
  // dataset.LoadMNIST();
  // int batch_size = 2;
  // hen::FloatTensor input_batches[60000/batch_size] {};  // { batch_size, 1, height, weight }
  // hen::FloatTensor label_batches[60000/batch_size] {};  // { batch_size }
  // dataset.GetTrainBatches(input_batches, label_batches, 60000/batch_size, batch_size);

  // // hen::FloatTensor input({2, 1, 28, 28});
  // hen::FloatTensor output({batch_size, 10});

  // // // draw number
  // // for (int i=1; i<=batch_size; i++) {
  // //   for (int h=1; h<=28; h++) {
  // //     for (int w=1; w<=28; w++) {
  // //       cout << setfill(' ') << setw(4);
  // //       cout << (int)input_batches[0].Get({i, 1, h, w});
  // //     }
  // //     cout << endl;
  // //   }
  // //   cout << endl << endl;
  // // }


  // hen::CNN model(2);
  // model.Forward(input_batches[0], output);

  // for (int b=1; b<=batch_size; b++) {
  //   for (int i=1; i<=10; i++) {
  //     cout << output.Get({b, i}) << " ";
  //   }
  //   cout << endl;
  // }
  
  // hen::CrossEntropyLoss cel;
  // float loss = cel.Loss(output, label_batches[0]);
  // cout << "loss: " << loss << endl;
  // model.Backward(input_batches[0], output);

  cout << "asd" << endl;

  hen::MNIST dataset("examples/mnist");
  dataset.LoadMNIST();
  int batch_size = 2;
  hen::FloatTensor input_batches[60000/batch_size] {};  // { batch_size, 1, height, weight }
  hen::FloatTensor label_batches[60000/batch_size] {};  // { batch_size }
  dataset.GetTrainBatches(input_batches, label_batches, 60000/batch_size, batch_size);

  hen::FloatTensor output({batch_size, 10});

  hen::CNN model(2);
  hen::CrossEntropyLoss cel;
  hen::SGD optimizer(0.01, 0.9);

  float loss = 0;
  int epoch = 3;
  for (int e=0; e<epoch; e++) {
    cout << "Epoch : " << e << endl;
    // train
    for (int batch=0; batch<5; batch++) {
      model.Forward(input_batches[batch], output);
      loss = cel.Loss(output, label_batches[batch]);
      cout << "loss: " << loss << endl;
      model.Backward(input_batches[batch], output);
      optimizer.Step(model);
      cout << "Epoch : " << e << " | batch : " << batch << "/60000 | loss : " << loss << endl;
    }
    // predict
  }






  return 0;
}
