#include <math.h>
#include <stdio.h>

#define PRINT_INTERMEDIATE_RESULTS

// define dtypes used in layers
typedef float din_t;
typedef float dout_t;

#define INPUT_DIM 2
#define SEQ_LEN 8
#define NUM_HEADS 2
#define HEAD_DIM 4
#define BLOCK_INPUT_DIM 8

// toplevel function
void transformer(din_t* input_array, dout_t* val);

dout_t transformer_for_sample(din_t xx[SEQ_LEN][INPUT_DIM]);

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

template <typename T, int T1, int T2, int T3>
void print_mat_3d(T mat[T1][T2][T3]) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      for (int k = 0; k < T3; k++) {
        printf("%.6f ", mat[i][j][k]);
      }
      printf("\n");
    }
    printf("\n--\n");
  }
}

template <typename T, int M, int N, int K>
void kernel_mmult(const T a[M][N], const T b[N][K], T out[M][K]) {
#pragma HLS ARRAY_PARTITION variable = a complete dim = 2
#pragma HLS ARRAY_PARTITION variable = b complete dim = 1
loop_i:
  for (int i = 0; i < M; ++i) {
#pragma HLS PIPELINE off
  loop_j:
    for (int j = 0; j < K; ++j) {
#pragma HLS PIPELINE
      T sum = 0;
    loop_k:
      for (int k = 0; k < N; ++k) {
#pragma HLS PIPELINE
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

// ELU function applied in place with the default alpha=1
template <typename T, int T1, int T2>
void elu_inplace(T A[T1][T2]) {
elu_inplace_loop_i:
  for (int i = 0; i < T1; i++) {
  elu_inplace_loop_j:
    for (int j = 0; j < T2; j++) {
      if (A[i][j] < 0) {
        A[i][j] = exp(A[i][j]) - 1;
      }
    }
  }
}

// relu inplace on 2d array
template <typename T, int T1, int T2>
void relu_inplace(T A[T1][T2]) {
relu_inplace_loop_i:
  for (int i = 0; i < T1; i++) {
#pragma HLS UNROLL
  relu_inplace_loop_j:
    for (int j = 0; j < T2; j++) {
#pragma HLS UNROLL
      if (A[i][j] < 0) {
        A[i][j] = 0;
      }
    }
  }
}

// helper function to transpose first two dimensions of a 3d array
// used for multihead attention
template <typename T, int T1, int T2, int T3>
void transpose_3d(const T A[T1][T2][T3], T result[T2][T1][T3]) {
transpose_3d_loop_i:
  for (int i = 0; i < T1; i++) {
  transpose_3d_loop_j:
    for (int j = 0; j < T2; j++) {
    transpose_3d_loop_k:
      for (int k = 0; k < T3; k++) {
        result[j][i][k] = A[i][j][k];
      }
    }
  }
}

// matrix transpose
// T1 is the number of rows, T2 is the number of columns of the input matrix
template <typename T, int T1, int T2>
void transpose(const T* A, T* result) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      result[j * T1 + i] = A[i * T2 + j];
    }
  }
}
