#include <iostream>
#include "hen.h"

using namespace std;

int main() {

  // user encrypt the data and send to cloud. 
  int data[2] = {1, 2};

  hen::User user = hen::User(2048, 256);

  std::vector<seal::Ciphertext> c_input;

  c_input = user.Encrypt(data, 2);

  // cloud compute and send result to the user.
  hen::Cloud cloud = hen::Cloud(user.GetUserPack());

  std::vector<seal::Ciphertext> c_output;

  c_output = cloud.Compute(c_input);

  // use decrypt the result.

  std::vector<int> res;

  res = user.Decrypt(c_output);

  cout << "1+2 = " << res[0] << endl;

  return 0;
}