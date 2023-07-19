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

// matrix multiplication A x B in a naive way
template <typename T, int T1, int T2, int T3>
void matmul(const T* A, const T* B, T* result);

// matrix transpose
template <typename T, int T1, int T2>
void transpose(const T* A, T* result);

template <typename T, int S>
void softmax(const T* A, T* result);

void attention(const dout_t query_mat[DIM_QUERY * INPUT_EMBEDDING_SIZE],
               const dout_t key_mat[DIM_KEY * INPUT_EMBEDDING_SIZE],
               const dout_t value_mat[DIM_VALUE * INPUT_EMBEDDING_SIZE],
               const din_t x[INPUT_LENGTH * INPUT_EMBEDDING_SIZE],
               dout_t output[INPUT_LENGTH * DIM_VALUE]);

// for debugging
void print_mat(const dout_t* mat, int T1, int T2);