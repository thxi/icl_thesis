#include <iostream>

// define static array of numbers from 0 to 9 of type double
// using the #define macro

#include "scaled_dp_params.h"
#include "transformer.h"

#define SEQ_LEN 8
#define NUM_HEADS 2
#define HEAD_DIM 4

int main() {
  using namespace std;
  // print the array
  static double xx[SEQ_LEN][2] = {
      {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 1},

  };
  double yy[SEQ_LEN * 8] = {0};
  // for every element in the sequence, apply front linear
  // front_linear is 8x2
  // xx is 8x2
  // so yy is 8x1
  for (int i = 0; i < SEQ_LEN; i++) {
    matmul<double, 8, 2, 1>(FRONT_LINEAR_WEIGHT, xx[i], yy + 8 * i);
    add_bias<double, 8>(yy + 8 * i, FRONT_LINEAR_BIAS);
  }
  cout << "yy: " << endl;
  print_mat(yy, SEQ_LEN, 8);

  // qkv proj
  double qkv[SEQ_LEN * 3 * 8] = {0};
  for (int i = 0; i < SEQ_LEN; i++) {
    matmul<double, 3 * 8, 8, 1>(
        TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_QKV_PROJ_WEIGHT, yy + 8 * i,
        qkv + 3 * 8 * i);
  }

  cout << "qkv: " << endl;
  print_mat(qkv, SEQ_LEN, 3 * 8);

  // separate query key and value

  // double q[SEQ_LEN * HEAD_DIM * NUM_HEADS] = {0};
  // double k[SEQ_LEN * HEAD_DIM * NUM_HEADS] = {0};
  // double v[SEQ_LEN * HEAD_DIM * NUM_HEADS] = {0};

  // for (int i = 0; i < SEQ_LEN; i++) {
  //   for (int j = 0; j < NUM_HEADS; j++) {
  //     for (int l = 0; l < HEAD_DIM; l++) {
  //       q[i * NUM_HEADS * HEAD_DIM + j * HEAD_DIM + l] =
  //           qkv[i * 3 * 8 + j * 8 + l];
  //       k[i * NUM_HEADS * HEAD_DIM + j * HEAD_DIM + l] =
  //           qkv[i * 3 * 8 + j * 8 + l + 4];
  //       v[i * NUM_HEADS * HEAD_DIM + j * HEAD_DIM + l] =
  //           qkv[i * 3 * 8 + j * 8 + l + 8];
  //     }
  //   }
  // }

  double qkv_reshaped[SEQ_LEN * 2 * 3 * 4] = {0};
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int i = 0; i < NUM_HEADS; ++i) {
      for (int j = 0; j < 3 * HEAD_DIM; ++j) {
        int flat_idx_qkv = (s * 2 * 3 * 4) + (i * 3 * 4) + j;
        int flat_idx_reshaped = (s * 2 * 3 * 4 * 2) + (i * 3 * 4) + j;

        // qkv_reshaped[s][i][j] = qkv[b][s][i * (3 * 4) + j];
        qkv_reshaped[flat_idx_reshaped] = qkv[flat_idx_qkv];
      }
    }
  }

  double q[SEQ_LEN * HEAD_DIM * NUM_HEADS] = {0};
  double k[SEQ_LEN * HEAD_DIM * NUM_HEADS] = {0};
  double v[SEQ_LEN * HEAD_DIM * NUM_HEADS] = {0};
  int chunk_size = 3;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int s = 0; s < SEQ_LEN; ++s) {
        for (int c = 0; c < chunk_size; ++c) {
          int flat_idx_chunked = (i * 8 * SEQ_LEN * chunk_size) +
                                 (j * SEQ_LEN * chunk_size) + (s * chunk_size) +
                                 c;
          int flat_idx_reshaped = (s * 2 * 3 * 4) + (i * 3 * 4) + (j * 3) + c;
          q[flat_idx_chunked] = qkv_reshaped[flat_idx_reshaped];
          k[flat_idx_chunked] = qkv_reshaped[flat_idx_reshaped + 3];
          v[flat_idx_chunked] = qkv_reshaped[flat_idx_reshaped + 2 * 3];
        }
      }
    }
  }

  cout << "q: " << endl;
  print_mat(q, SEQ_LEN, 8);
  cout << "k: " << endl;
  print_mat(k, SEQ_LEN, 8);
  cout << "v: " << endl;
  print_mat(v, SEQ_LEN, 8);
  return 0;
}