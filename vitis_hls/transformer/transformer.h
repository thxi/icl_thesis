// attention layer implemented in c++
#include <math.h>

// define dtypes used in layers
typedef double din_t;
typedef double dout_t;

// define dimensions
#define INPUT_LENGTH 6          // number of observations in input sentence
#define INPUT_EMBEDDING_SIZE 4  // embedding dimension of the input sequence

#define DIM_QUERY 5
#define DIM_KEY 5  // have to be the same as DIM_QUERY
#define DIM_VALUE 7
// so that the Query, Key and Value matrices are
// DIM_QUERY x INPUT_EMBEDDING_SIZE, etc.

void attention(const dout_t query_mat[DIM_QUERY * INPUT_EMBEDDING_SIZE],
               const dout_t key_mat[DIM_KEY * INPUT_EMBEDDING_SIZE],
               const dout_t value_mat[DIM_VALUE * INPUT_EMBEDDING_SIZE],
               const din_t x[INPUT_LENGTH * INPUT_EMBEDDING_SIZE],
               dout_t output[INPUT_LENGTH * DIM_VALUE]);

// for debugging
void print_mat(const dout_t* mat, int T1, int T2);

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
  T sum = 0;
  for (int i = 0; i < S; i++) {
#pragma HLS pipeline off
    sum += exp(A[i]);
  }
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

template <typename T, int T1>
void add_bias(T* A, const T* bias) {
  for (int i = 0; i < T1; i++) {
#pragma HLS pipeline off
    A[i] += bias[i];
  }
}