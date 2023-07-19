#include "ap_axi_sdata.h"
#include "hls_stream.h"

// define a static matrix of size S1xS2
#define S1 6
#define S2 5
#define S3 4
int query_matrix[S1 * S2] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

// just taking pointers to gmem
// https://github.com/Xilinx/Vitis_Accel_Examples/blob/63bae10d581df40cf9402ed71ea825476751305d/hello_world/src/vadd.cpp

// sizes of matrices:
// the query matrix S1xS2 (on the board directly)
// the input matrix is S2xS3

// it is multiplied by the query matrix
void toplevelmatmul(const int* in_mat, int* result) {
#pragma HLS INTERFACE m_axi port = in_mat offset = slave bundle = gmem
#pragma HLS INTERFACE m_axi port = result offset = slave bundle = gmem
#pragma HLS INTERFACE s_axilite port = in_mat bundle = control
#pragma HLS INTERFACE s_axilite port = result bundle = control
#pragma HLS INTERFACE s_axilite port = return

// multiply query_matrix by in_mat and write to result
loop1:
  for (int i = 1; i < S1; i++) {
  loop2:
    for (int j = 0; j < S3; j++) {
      int sum = 0;
    loop3:
      for (int k = 0; k < S2; k++) {
#pragma HLS pipeline off
        sum += query_matrix[i * S2 + k] * in_mat[k * S3 + j];
      }
      result[i * S3 + j] = sum;
    }
  }
}