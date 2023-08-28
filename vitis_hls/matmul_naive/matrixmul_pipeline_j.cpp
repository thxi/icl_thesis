#include "matrixmul.h"

void matmul(mat_a_t a[MAT_A_ROWS][MAT_A_COLS],
            mat_b_t b[MAT_B_ROWS][MAT_B_COLS],
            result_t res[MAT_A_ROWS][MAT_B_COLS]) {
#pragma HLS ARRAY_PARTITION variable = a complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b complete dim = 1
loop_i:
  for (int i = 0; i < MAT_A_ROWS; i++) {
#pragma HLS PIPELINE off
  loop_j:
    for (int j = 0; j < MAT_B_COLS; j++) {
      int tmp = 0;
#pragma HLS PIPELINE
    loop_k:
      for (int k = 0; k < MAT_B_ROWS; k++) {
#pragma HLS UNROLL
        tmp += a[i][k] * b[k][j];
      }
      res[i][j] = tmp;
    }
  }
}
