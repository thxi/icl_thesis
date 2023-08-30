#include <math.h>
#include <stdio.h>

// define dtypes used in layers
typedef double din_t;
typedef double dout_t;

#define INPUT_DIM 2
#define SEQ_LEN 8
#define NUM_HEADS 2
#define HEAD_DIM 4
#define BLOCK_INPUT_DIM 8

// toplevel function
dout_t transformer(din_t xx[SEQ_LEN][INPUT_DIM]);

// for debugging
void print_mat(const dout_t* mat, int T1, int T2);

template <typename T, int T1, int T2>
void print_mat_template(const T mat[T1][T2]) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      printf("%.6f ", mat[i][j]);
    }
    printf("\n");
  }
}

// template <typename T, int M, int N, int K>
// void kernel_mmult(T a[M * N], T b[N * K], T out[M * K]) {
// loop_i:
//   for (int i = 0; i < M; ++i) {
// #pragma HLS PIPELINE off
//   loop_j:
//     for (int j = 0; j < N; ++j) {
// #pragma HLS PIPELINE
//       T sum = 0;
//     loop_k:
//       for (int k = 0; k < K; ++k) {
// #pragma HLS UNROLL
//         sum += a[i * N + k] * b[k * N + j];
//       }
//       out[i * M + j] = sum;
//     }
//   }
//   return;
// }

template <typename T, int M, int N, int K>
void kernel_mmult(const T a[M][N], const T b[N][K], T out[M][K]) {
loop_i:
  for (int i = 0; i < M; ++i) {
#pragma HLS PIPELINE off
  loop_j:
    for (int j = 0; j < K; ++j) {
#pragma HLS PIPELINE
      T sum = 0;
    loop_k:
      for (int k = 0; k < N; ++k) {
#pragma HLS UNROLL
        sum += a[i][k] * b[k][j];
      }
      out[i][j] = sum;
    }
  }
  return;
}

// basically, vector add
template <typename T, int T1>
void add_bias_to_row(const T* A, const T* bias, T* out) {
  for (int i = 0; i < T1; i++) {
    out[i] = A[i] + bias[i];
  }
}

template <typename T, int M, int N>
void kernel_bias_add(const T a[M][N], const T bias[N], T out[M][N]) {
  for (int i = 0; i < M; i++) {
#pragma HLS pipeline
    add_bias_to_row<dout_t, N>(a[i], bias, out[i]);
  }
}

// matrix multiplication A x B in a naive way
template <typename T, int T1, int T2, int T3>
void matmul(const T* A, const T* B, T* result) {
  for (int i = 0; i < T1; i++) {
#pragma HLS pipeline off
    for (int j = 0; j < T3; j++) {
#pragma HLS pipeline off
      T sum = 0;
      for (int k = 0; k < T2; k++) {
#pragma HLS pipeline off
        sum += A[i * T2 + k] * B[k * T3 + j];
      }
      result[i * T3 + j] = sum;
    }
  }
}

template <typename T, int S>
void softmax(const T* A, T* result) {
  // numerical stability
  T sum = 0;
  for (int i = 0; i < S; i++) {
#pragma HLS pipeline off
    sum += exp(A[i]);
  }
  printf("sum: %.6f\n", sum);
  for (int i = 0; i < S; i++) {
#pragma HLS pipeline off
    result[i] = exp(A[i]) / sum;
  }
}

// matrix transpose
// T1 is the number of rows, T2 is the number of columns of the input matrix
template <typename T, int T1, int T2>
void transpose(const T* A, T* result) {
  for (int i = 0; i < T1; i++) {
#pragma HLS pipeline off
    for (int j = 0; j < T2; j++) {
#pragma HLS pipeline off
      result[j * T1 + i] = A[i * T2 + j];
    }
  }
}
