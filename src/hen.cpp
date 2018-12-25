// #include <seal/seal.h>
// #include "hen.h"

// namespace hen {
//   std::vector<seal::Ciphertext> encrypt(int poly_modulus, int plain_modulus, std::vector<int> input) {
//     // set parameters
//     seal::EncryptionParameters params(seal::scheme_type::BFV);
//     params.set_poly_modulus_degree(poly_modulus);
//     params.set_coeff_modulus(seal::coeff_modulus_128(poly_modulus));
//     params.set_plain_modulus(plain_modulus);
//     // construct a SEALContext object
//     std::shared_ptr<seal::SEALContext> context = seal::SEALContext::Create(params);
//     // encoder use to encode input integer
//     seal::IntegerEncoder encoder(params.plain_modulus());  // only encode int
//     // generator keys
//     seal::KeyGenerator keygen(context);
//     seal::PublicKey public_key = keygen.public_key();
//     seal::SecretKey secret_key = keygen.secret_key();
//     // encryptor and decryptor
//     seal::Encryptor encryptor(context, public_key);
//     seal::Decryptor decryptor(context, secret_key);

//     // encrypt

//   }

//   std::vector<int> decrypt(int poly_modulus, int plain_modulus, std::vector<int> input) {
//     // set parameters
//     seal::EncryptionParameters params(seal::scheme_type::BFV);
//     params.set_poly_modulus_degree(poly_modulus);
//     params.set_coeff_modulus(seal::coeff_modulus_128(poly_modulus));
//     params.set_plain_modulus(plain_modulus);
//     // construct a SEALContext object
//     std::shared_ptr<seal::SEALContext> context = seal::SEALContext::Create(params);
//     // encoder use to encode input integer
//     seal::IntegerEncoder encoder(params.plain_modulus());  // only encode int
//     // generator keys
//     seal::KeyGenerator keygen(context);
//     seal::PublicKey public_key = keygen.public_key();
//     seal::SecretKey secret_key = keygen.secret_key();
//     // encryptor and decryptor
//     // seal::Encryptor encryptor(context, public_key);
//     seal::Decryptor decryptor(context, secret_key);
//   }

//   std::vector<seal::Ciphertext> compute(int poly_modulus, int plain_modulus, std::vector<seal::Ciphertext> cinput) {

//   }

// }