#include "transformer.h"

#include <stdio.h>

#include <iostream>

#include "scaled_dp_params.h"

void print_mat(const dout_t* mat, int T1, int T2) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      printf("%.6f ", mat[i * T2 + j]);
    }
    printf("\n");
  }
}

#define PRINT_INTERMEDIATE_RESULTS

using namespace std;

dout_t transformer(din_t xx[SEQ_LEN][INPUT_DIM]) {
// for every element in the sequence, apply front linear
// front_linear is 8x2 but it is stored as a transposed matrix
// so multiply on the right
// xx is 8x2
// so yy is 8x8
front_linear:
  dout_t front_linear_output[SEQ_LEN][BLOCK_INPUT_DIM] = {0};

  kernel_mmult<dout_t, SEQ_LEN, INPUT_DIM, BLOCK_INPUT_DIM>(
      xx, FRONT_LINEAR_WEIGHT, front_linear_output);
  kernel_bias_add<dout_t, SEQ_LEN, BLOCK_INPUT_DIM>(
      front_linear_output, FRONT_LINEAR_BIAS, front_linear_output);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "front_linear_output: " << endl;
  print_mat((dout_t*)front_linear_output, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  // Transformer encoder
  dout_t query_mat[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  dout_t key_mat[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  dout_t value_mat[SEQ_LEN][BLOCK_INPUT_DIM] = {0};

query_mat:
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      front_linear_output,
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_QUERY_MAT_WEIGHT, query_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "query_mat: " << endl;
  print_mat((dout_t*)query_mat, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

key_mat:
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      front_linear_output,
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_KEY_MAT_WEIGHT, key_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "key_mat: " << endl;
  print_mat((dout_t*)key_mat, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

value_mat:
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      front_linear_output,
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_VALUE_MAT_WEIGHT, value_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "value_mat: " << endl;
  print_mat((dout_t*)value_mat, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  return query_mat[0][0];
}