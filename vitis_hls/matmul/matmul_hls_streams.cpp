#include "ap_axi_sdata.h"
#include "hls_stream.h"

// define a static matrix of size S1xS2
#define S1 6
#define S2 5
#define S3 4
int query_matrix[S1 * S2] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29};

// similar to streaming without side channel
// https://github.com/Xilinx/Vitis-HLS-Introductory-Examples/blob/2021.2/Interface/Streaming/using_axis_array_stream_no_side_channel_data/example.cpp

// but taking the fixed length streams:
// https://support.xilinx.com/s/question/0D52E00006hplnISAQ/hls-stream-apaxis?language=en_US

// sizes of matrices:
// the query matrix S1xS2 (on the board directly)
// the input matrix is S2xS3

// it is multiplied by the query matrix
void toplevelmatmul(hls::stream<ap_axis<32, 2, 5, 6>> &in_mat,
                    hls::stream<ap_axis<32, 2, 5, 6>> &result) {
#pragma HLS INTERFACE axis port = in_mat
#pragma HLS INTERFACE axis port = result
#pragma HLS INTERFACE s_axilite port = return

  // multiply query_matrix by in_mat and write to result
  ap_axis<32, 2, 5, 6> tmp;
  for (int i = 0; i < S1; i++) {
    for (int j = 0; j < S3; j++) {
      int sum = 0;
      for (int k = 0; k < S2; k++) {
        in_mat.read(tmp);
        sum += query_matrix[i * S2 + k] * tmp.data.to_int();
      }
      tmp.data = sum;
      result.write(tmp);
      if (tmp.last) {
        return;
      }
    }
  }
}
