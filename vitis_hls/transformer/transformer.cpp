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

dout_t transformer(din_t xx[SEQ_LEN][INPUT_LEN]) {
  // for every element in the sequence, apply front linear
  // front_linear is 8x2 but it is stored as a transposed matrix
  // so multiply on the right
  // xx is 8x2
  // so yy is 8x8
  dout_t front_linear_output[SEQ_LEN][8] = {0};

  kernel_mmult<dout_t, 8, 2, 8>(xx, FRONT_LINEAR_WEIGHT, front_linear_output);
  kernel_bias_add<dout_t, 8, 8>(front_linear_output, FRONT_LINEAR_BIAS,
                                front_linear_output);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "front_linear_output: " << endl;
  print_mat((dout_t*)front_linear_output, SEQ_LEN, 8);
#endif

  return 0;
}