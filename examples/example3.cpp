#include <ctime>
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

  // cout << "asd" << endl;

  hen::MNIST dataset("examples/mnist");
  dataset.LoadMNIST();
  // int batch_size = 1;
  int batch_size = 128;

  hen::FloatTensor train_input_batches[60000/batch_size] {};  // { batch_size, 1, height, weight }
  hen::FloatTensor train_label_batches[60000/batch_size] {};  // { batch_size }
  dataset.GetTrainBatches(train_input_batches, train_label_batches, 60000/batch_size, batch_size);

  hen::FloatTensor test_input_batches[60000/batch_size] {};  // { batch_size, 1, height, weight }
  hen::FloatTensor test_label_batches[60000/batch_size] {};  // { batch_size }
  dataset.GetTestBatches(test_input_batches, test_label_batches, 10000/batch_size, batch_size);

  hen::FloatTensor output({batch_size, 10});

  // hen::CNN model(batch_size);
  // hen::MLP model(batch_size);
  hen::NN1 model(batch_size);
  hen::CrossEntropyLoss cel;
  hen::SGD optimizer(0.01, 0.5);

  int batches = 60000 / batch_size;
  float loss = 0;
  int epoch = 3;
  clock_t start_time = 0;
  cout << "Batch Size : " << batch_size << endl;
  for (int e=0; e<epoch; e++) {
    // train
    start_time = clock();
    for (int batch=0; batch<batches; batch++) {
      // hen::PrintSample(input_batches[batch]);
      model.Forward(train_input_batches[batch], output);
      loss = cel.Loss(output, train_label_batches[batch]);
      model.Backward(train_input_batches[batch], output);
      optimizer.Step(model);
      if (batch % 20 == 0) {
        cout << "Train Epoch : " << e << " | batch : " << batch << "/" << batches << " | loss : " << loss << endl;
      }
    }
    cout << "Time : " << (float)(clock() - start_time) / CLOCKS_PER_SEC << " s" << endl; 
    // predict
    start_time = clock();
    int correct = 0;
    for (int batch=0; batch<10000/batch_size; batch++) {
      model.Forward(test_input_batches[batch], output);
      correct += hen::CountCorrect(output, test_label_batches[batch]);
    }
    cout << endl;
    cout << "Test time : " << (float)(clock() - start_time) / CLOCKS_PER_SEC << "s  avg acc: " << correct << "/10000 (" << (float)correct/10000 << "%)" << endl;
    cout << endl;
  }






  return 0;
}
