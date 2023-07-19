#include "attention.h"

#include <cstdio>

void print_mat(const dout_t* mat, int T1, int T2) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      printf("%.6f ", mat[i * T2 + j]);
    }
    printf("\n");
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

void attention(const dout_t query_mat[DIM_QUERY * INPUT_EMBEDDING_SIZE],
               const dout_t key_mat[DIM_KEY * INPUT_EMBEDDING_SIZE],
               const dout_t value_mat[DIM_VALUE * INPUT_EMBEDDING_SIZE],
               const din_t x[INPUT_LENGTH * INPUT_EMBEDDING_SIZE],
               dout_t output[INPUT_LENGTH * DIM_VALUE]) {
  // compute query, key and values for each observation
  dout_t queries[INPUT_LENGTH * DIM_QUERY];
  dout_t keys[INPUT_LENGTH * DIM_KEY];
  dout_t values[INPUT_LENGTH * DIM_VALUE];

  // TODO: transposes take time, just compute products directly
  // matmul<dout_t, INPUT_LENGTH, INPUT_EMBEDDING_SIZE, DIM_QUERY>(x, query_mat,
  //                                                               queries);

  // transposed matrices for the product
  din_t x_t[INPUT_EMBEDDING_SIZE * INPUT_LENGTH];
  dout_t queries_t[INPUT_LENGTH * DIM_QUERY];
  dout_t keys_t[INPUT_LENGTH * DIM_KEY];
  dout_t values_t[INPUT_LENGTH * DIM_VALUE];

  transpose<din_t, INPUT_LENGTH, INPUT_EMBEDDING_SIZE>(x, x_t);

  matmul<dout_t, DIM_QUERY, INPUT_EMBEDDING_SIZE, INPUT_LENGTH>(query_mat, x_t,
                                                                queries_t);
  matmul<dout_t, DIM_KEY, INPUT_EMBEDDING_SIZE, INPUT_LENGTH>(key_mat, x_t,
                                                              keys_t);
  matmul<dout_t, DIM_VALUE, INPUT_EMBEDDING_SIZE, INPUT_LENGTH>(value_mat, x_t,
                                                                values_t);

  transpose<dout_t, DIM_QUERY, INPUT_LENGTH>(queries_t, queries);
  // TODO: keys is not used
  transpose<dout_t, DIM_KEY, INPUT_LENGTH>(keys_t, keys);
  transpose<dout_t, DIM_VALUE, INPUT_LENGTH>(values_t, values);

  // compute the attention weights
  dout_t unnormalized_attention_weights[INPUT_LENGTH * INPUT_LENGTH];
  matmul<dout_t, INPUT_LENGTH, DIM_QUERY, INPUT_LENGTH>(
      queries, keys_t, unnormalized_attention_weights);

  for (int i = 0; i < INPUT_LENGTH; i++) {
    for (int j = 0; j < INPUT_LENGTH; j++) {
#pragma HLS pipeline off
      unnormalized_attention_weights[i * INPUT_LENGTH + j] /= sqrt(DIM_QUERY);
    }
  }

  dout_t attention_weights[INPUT_LENGTH * INPUT_LENGTH];
  for (int i = 0; i < INPUT_LENGTH; i++) {
#pragma HLS pipeline off
    softmax<dout_t, INPUT_LENGTH>(
        unnormalized_attention_weights + i * INPUT_LENGTH,
        attention_weights + i * INPUT_LENGTH);
  }

  // compute the output
  matmul<dout_t, INPUT_LENGTH, INPUT_LENGTH, DIM_VALUE>(attention_weights,
                                                        values, output);
}
