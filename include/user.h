#pragma once

#include <seal/seal.h>
#include "pack.h"
#include "tensor.h"

namespace hen {

// @class User
class User {
  public:
    User(int poly_modulus, int plain_modulus);
    // ~User();
    std::vector<seal::Ciphertext> Encrypt(int* data, int length);
    std::vector<int> Decrypt(std::vector<seal::Ciphertext> data);

    void Encrypt(IntTensor &plain_tensor, CipherTensor &cipher_tenosr);
    void Decrypt(CipherTensor &cipher_tensor, IntTensor &plain_tenosr);

    seal::PublicKey GetPublicKey();
    Pack GetUserPack();
    static seal::EncryptionParameters param(int poly_modulus, int plain_modulus);

  private:
    std::shared_ptr<seal::SEALContext> context_;
    seal::IntegerEncoder encoder_;
    seal::KeyGenerator keygen_;
    seal::PublicKey public_key_;
    seal::SecretKey secret_key_;
    seal::Encryptor encryptor_;
    seal::Decryptor decryptor_;
};

} // namespace hen