#include "matmul.h"

// similar to streaming without side channel
// https://github.com/Xilinx/Vitis-HLS-Introductory-Examples/blob/2021.2/Interface/Streaming/using_axis_array_stream_no_side_channel_data/example.cpp

// sizes of matrices:
// the query matrix S1xS2 (on the board directly)
// the input matrix is S2xS3

// it is multiplied by the query matrix
void toplevelmatmul(int in_mat[S2 * S3], int result[S1 * S3]) {
#pragma HLS INTERFACE axis port = in_mat
#pragma HLS INTERFACE axis port = result
// TODO: maybe not needed? because we don't have a side channel
#pragma hls interface s_axilite port = return

  mul_mat_mat<int, S1, S2, S3>(query_matrix, in_mat, result);
}

// second try
#include "matmul.h"

// similar to streaming without side channel
// https://github.com/Xilinx/Vitis-HLS-Introductory-Examples/blob/2021.2/Interface/Streaming/using_axis_array_stream_no_side_channel_data/example.cpp

// sizes of matrices:
// the query matrix S1xS2 (on the board directly)
// the input matrix is S2xS3

// it is multiplied by the query matrix
void toplevelmatmul(int in_mat[S2 * S3], int result[S1 * S3]) {
#pragma HLS INTERFACE axis port = in_mat
#pragma HLS INTERFACE axis port = result

  mul_mat_mat<int, S1, S2, S3>(query_matrix, in_mat, result);
}
