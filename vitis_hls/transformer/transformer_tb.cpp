#include <limits.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <string>

#include "transformer.h"

int main() {
  dout_t query_mat[DIM_QUERY * INPUT_EMBEDDING_SIZE];
  std::ifstream query_mat_file("W_query");
  for (int i = 0; i < DIM_QUERY * INPUT_EMBEDDING_SIZE; i++) {
    query_mat_file >> query_mat[i];
  }

  dout_t key_mat[DIM_KEY * INPUT_EMBEDDING_SIZE];
  std::ifstream key_mat_file("W_key");
  for (int i = 0; i < DIM_KEY * INPUT_EMBEDDING_SIZE; i++) {
    key_mat_file >> key_mat[i];
  }

  dout_t value_mat[DIM_VALUE * INPUT_EMBEDDING_SIZE];
  std::ifstream value_mat_file("W_value");
  for (int i = 0; i < DIM_VALUE * INPUT_EMBEDDING_SIZE; i++) {
    value_mat_file >> value_mat[i];
  }

  din_t x[INPUT_LENGTH * INPUT_EMBEDDING_SIZE];
  std::ifstream x_file("x");
  for (int i = 0; i < INPUT_LENGTH * INPUT_EMBEDDING_SIZE; i++) {
    x_file >> x[i];
  }

  dout_t expected_output[INPUT_LENGTH * DIM_VALUE];
  std::ifstream expected_output_file("z");
  for (int i = 0; i < INPUT_LENGTH * DIM_VALUE; i++) {
    expected_output_file >> expected_output[i];
  }

  dout_t output[INPUT_LENGTH * DIM_VALUE];

  attention(query_mat, key_mat, value_mat, x, output);

  printf("output:\n");
  print_mat(output, INPUT_LENGTH, DIM_VALUE);

  // compare to expected
  bool success = true;
  for (int i = 0; i < INPUT_LENGTH * DIM_VALUE; i++) {
    if (abs(output[i] - expected_output[i]) > 1e-5) {
      success = false;
      printf("output[%d] = %.6f, expected_output[%d] = %.6f\n", i, output[i], i,
             expected_output[i]);
    }
  }
  return success ? 0 : 1;
}