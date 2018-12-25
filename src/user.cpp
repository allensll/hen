#include <ctime>
#include <seal/seal.h>
#include "user.h"

namespace hen {
// @class User
seal::EncryptionParameters User::param(int poly_modulus, int plain_modulus) {
  // set parameters
  seal::EncryptionParameters params(seal::scheme_type::BFV);
  params.set_poly_modulus_degree(poly_modulus);
  params.set_coeff_modulus(seal::coeff_modulus_128(poly_modulus));
  params.set_plain_modulus(plain_modulus);
  return params;
}

User::User(int poly_modulus, int plain_modulus) :
  context_(seal::SEALContext::Create(User::param(poly_modulus, plain_modulus))),
  encoder_(User::param(poly_modulus, plain_modulus).plain_modulus()),
  keygen_(context_),
  public_key_(keygen_.public_key()),
  secret_key_(keygen_.secret_key()),
  encryptor_(context_, public_key_),
  decryptor_(context_, secret_key_)
{
  // // set parameters
  // seal::EncryptionParameters params(seal::scheme_type::BFV);
  // params.set_poly_modulus_degree(poly_modulus);
  // params.set_coeff_modulus(seal::coeff_modulus_128(poly_modulus));
  // params.set_plain_modulus(plain_modulus);
  // // construct a SEALContext object
  // context_ = seal::SEALContext::Create(params);
  // // encoder use to encode input integer
  // encoder_ = &seal::IntegerEncoder(params.plain_modulus());  // only encode int
  // // generator keys
  // keygen_ = &seal::KeyGenerator(context_);
  // public_key_ = keygen_->public_key();
  // secret_key_ = keygen_->secret_key();
  // // encryptor and decryptor
  // encryptor_ = &seal::Encryptor(context_, public_key_);
  // decryptor_ = &seal::Decryptor(context_, secret_key_);
}

// User::~User() {
// }

seal::PublicKey User::GetPublicKey() {
  return public_key_;
}

std::vector<seal::Ciphertext> User::Encrypt(int data[], int length) {
  std::vector<seal::Ciphertext> encrypted;
  int temp;
  for (int i = 0; i < length; i++) {
    seal::Plaintext pt = encoder_.encode(data[i]);
    seal::Ciphertext ct;
    encryptor_.encrypt(pt, ct);
    encrypted.insert(encrypted.end(), ct);
  }
  return encrypted;
}

std::vector<int> User::Decrypt(std::vector<seal::Ciphertext> data) {
  std::vector<int> decrypted;
  for (seal::Ciphertext ctext : data) {
    seal::Plaintext ptext;
    decryptor_.decrypt(ctext, ptext);
    decrypted.insert(decrypted.end(), encoder_.decode_int32(ptext));
  }
  return decrypted;
}

void User::Encrypt(IntTensor &plain_tensor, CipherTensor &cipher_tensor) {
  // seal::Plaintext pt;
  clock_t start_time = clock();
  seal::Ciphertext ct;
  for (int i = 0; i < plain_tensor.data_size_; i++) {
    // if (i % 1000 == 0)
    //   std::cout << i << std::endl;
    // pt = encoder_.encode(plain_tensor.data_[i]);
    encryptor_.encrypt(encoder_.encode(plain_tensor.data_[i]), ct);
    cipher_tensor.data_[i] = ct;
  }
  std::cout << "encrypt time is " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;
}

void User::Decrypt(CipherTensor &cipher_tensor, IntTensor &plain_tensor) {
  clock_t start_time = clock();
  seal::Plaintext ptext;
  for (int i = 0; i < cipher_tensor.data_size_; i++) {
    decryptor_.decrypt(cipher_tensor.data_[i], ptext);
    plain_tensor.data_[i] = encoder_.decode_int32(ptext);
  }
  std::cout << "decrypt time is " << (double)(clock() - start_time) / CLOCKS_PER_SEC << " s" << std::endl;
}



Pack User::GetUserPack() {
  return Pack(context_, public_key_);
}


} // namespace hen