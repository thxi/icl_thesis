#include "matrixmul.h"

void matmul(mat_a_t a[MAT_A_ROWS][MAT_A_COLS],
            mat_b_t b[MAT_B_ROWS][MAT_B_COLS],
            result_t res[MAT_A_ROWS][MAT_B_COLS]) {
  int temp_sum[MAT_B_COLS];
#pragma HLS ARRAY_PARTITION variable = b dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = res dim = 2 complete
#pragma HLS ARRAY_PARTITION variable = temp_sum dim = 1 complete
loop_i:
  for (int i = 0; i < MAT_A_ROWS; i++) {
  loop_k:
    for (int k = 0; k < MAT_B_ROWS; k++) {
#pragma HLS PIPELINE
    loop_j:
      for (int j = 0; j < MAT_B_COLS; j++) {
#pragma HLS UNROLL
        int result = (k == 0) ? 0 : temp_sum[j];
        result += a[i][k] * b[k][j];
        temp_sum[j] = result;
        if (k == MAT_B_ROWS - 1) {
          res[i][j] = result;
        }
      }
    }
  }
}
