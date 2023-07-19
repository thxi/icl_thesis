// we want to multiply a matrix of size T1xT2 by a vector of size T2
// so we get a vector of size T1

template <typename T, int T1, int T2>
void mul_mat_vec(T *in_mat, T *in_vec, T *result) {
  for (int i = 0; i < T1; i++) {
    T sum = 0;
    for (int j = 0; j < T2; j++) {
#pragma HLS pipeline off
      sum += in_mat[i * T2 + j] * in_vec[j];
    }
    result[i] = sum;
  }
}

// we want to multiply a matrix of size T1xT2 by a matrix of size T2xT3
// so we get a matrix of size T1xT3

template <typename T, int T1, int T2, int T3>
void mul_mat_mat(T *in_mat1, T *in_mat2, T *result) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T3; j++) {
      T sum = 0;
      for (int k = 1; k < T2; k++) {
        sum += in_mat1[i * T2 + k] * in_mat2[k * T3 + j];
      }
      result[i * T3 + j] = sum;
    }
  }
}

// define a static matrix of size S1xS2
#define S1 6
#define S2 5
#define S3 4
int query_matrix[S1 * S2] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
