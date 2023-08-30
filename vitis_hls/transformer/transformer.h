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

template <typename T, int S>
void softmax(T* A, T* result) {
  // numerical stability
  T max_element = A[0];
softmax_max_element:
  for (int i = 1; i < S; i++) {
    if (A[i] > max_element) {
      max_element = A[i];
    }
  }

softmax_subtract_max:
  for (int i = 0; i < S; i++) {
    A[i] = A[i] - max_element;
  }

  T sum = 0;
#ifdef PRINT_INTERMEDIATE_RESULTS
  printf("A: ");
#endif
softmax_sum:
  for (int i = 0; i < S; i++) {
#ifdef PRINT_INTERMEDIATE_RESULTS
    printf("%.6f;", A[i]);
#endif
    sum += exp(A[i]);
  }
#ifdef PRINT_INTERMEDIATE_RESULTS
  printf("\n");
#endif

#ifdef PRINT_INTERMEDIATE_RESULTS
  printf("max_element: %.6f sum: %.6f\n", max_element, sum);
#endif
softmax_calc_result:
  for (int i = 0; i < S; i++) {
    result[i] = exp(A[i]) / sum;
  }
}

// multi-head self-attention
template <typename T, int S, int H, int D>
void multihead_attention(T query[H][S][D], T key[H][S][D], T value[H][S][D],
                         T result[H][S][D]) {
  // query, key, value are all H x S x D
  // result is H x S x D

  // transpose last 2 dimensions of key

multihead_attention_key_transpose:
  T key_transposed[H][D][S] = {0};
  for (int i = 0; i < H; i++) {
    for (int j = 0; j < D; j++) {
      for (int k = 0; k < S; k++) {
        key_transposed[i][j][k] = key[i][k][j];
      }
    }
  }

  // calculate query * key^T for each head
multihead_attention_query_key:
  T query_key[H][S][S] = {0};
  for (int h = 0; h < H; h++) {
    // TODO: maybe roll back as not enough resources
#pragma HLS unroll
    kernel_mmult<T, S, D, S>(query[h], key_transposed[h], query_key[h]);
  }

  // scale by 1/sqrt(D)
multihead_attention_scale:
  T sqrtD = sqrt(D);
  for (int h = 0; h < H; h++) {
    for (int i = 0; i < S; i++) {
      for (int j = 0; j < S; j++) {
        query_key[h][i][j] = query_key[h][i][j] / sqrtD;
      }
    }
  }

  // apply softmax
multihead_attention_softmax:
  T query_key_softmax[H][S][S] = {0};
  for (int h = 0; h < H; h++) {
#pragma HLS PIPELINE off
    for (int i = 0; i < S; i++) {
#pragma HLS PIPELINE off
      softmax<T, S>(query_key[h][i], query_key_softmax[h][i]);
    }
  }

  // calculate query_key_softmax * value for each head
  // which is the result
multihead_attention_query_key_softmax_value:
  for (int h = 0; h < H; h++) {
    kernel_mmult<T, S, S, D>(query_key_softmax[h], value[h], result[h]);
  }
}

// helper function to split head into a multihead array
template <typename T, int seq_len, int num_heads, int head_dim>
void split_head(const T A[seq_len][num_heads * head_dim],
                T result[seq_len][num_heads][head_dim]) {
split_head_loop_i:
  for (int i = 0; i < seq_len; i++) {
  split_head_loop_j:
    for (int j = 0; j < num_heads; j++) {
    split_head_loop_k:
      for (int k = 0; k < head_dim; k++) {
        result[i][j][k] = A[i][j * head_dim + k];
      }
    }
  }
}

// helper function to concatenate heads into a single head
// inverse of split_head
template <typename T, int seq_len, int num_heads, int head_dim>
void concat_head(const T A[seq_len][num_heads][head_dim],
                 T result[seq_len][num_heads * head_dim]) {
concat_head_loop_i:
  for (int i = 0; i < seq_len; i++) {
  concat_head_loop_j:
    for (int j = 0; j < num_heads; j++) {
    concat_head_loop_k:
      for (int k = 0; k < head_dim; k++) {
        result[i][j * head_dim + k] = A[i][j][k];
      }
    }
  }
}

// relu inplace on 2d array
template <typename T, int T1, int T2>
void relu_inplace(T A[T1][T2]) {
relu_inplace_loop_i:
  for (int i = 0; i < T1; i++) {
#pragma HLS pipeline off
  relu_inplace_loop_j:
    for (int j = 0; j < T2; j++) {
#pragma HLS pipeline off
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
#pragma HLS pipeline off
    for (int j = 0; j < T2; j++) {
#pragma HLS pipeline off
      result[j * T1 + i] = A[i * T2 + j];
    }
  }
}
