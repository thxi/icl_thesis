#include <cstdio>
#include <iostream>

#include "matmul.h"

void print_mat(int* mat, int T1, int T2) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      printf("%d ", mat[i * T2 + j]);
    }
    printf("\n");
  }
}

void print_vec(int* vec, int T2) {
  for (int i = 0; i < T2; i++) {
    printf("%d ", vec[i]);
  }
  printf("\n");
}

int main() {
  printf("Hello World!\n");
  print_mat(query_matrix, S1, S2);
  {
    // check mat vec multiplication
    const int T1 = 4;
    const int T2 = 3;
    int* mat = new int[T1 * T2];
    // initialize mat
    for (int i = 0; i < T1; i++) {
      for (int j = 0; j < T2; j++) {
        mat[i * T2 + j] = i * T2 + j;
      }
    }
    printf("mat:\n");
    print_mat(mat, T1, T2);

    int* vec = new int[T2];
    // initialize vec
    for (int i = 0; i < T2; i++) {
      vec[i] = i + 1;
    }
    printf("vec:\n");
    print_vec(vec, T2);

    int* result = new int[T1];
    // set result to 0
    for (int i = 0; i < T1; i++) {
      result[i] = 0;
    }

    mul_mat_vec<int, T1, T2>(mat, vec, result);

    printf("result:\n");
    print_vec(result, T1);
    // should be 8 26 44 62
  }
  {
    // check mat mat multiplication
    const int T1 = 4;
    const int T2 = 3;
    const int T3 = 2;
    int* mat1 = new int[T1 * T2];
    // initialize mat1
    for (int i = 0; i < T1; i++) {
      for (int j = 0; j < T2; j++) {
        mat1[i * T2 + j] = i * T2 + j;
      }
    }
    printf("mat1:\n");
    print_mat(mat1, T1, T2);

    int* mat2 = new int[T2 * T3];
    // initialize mat2
    for (int i = 0; i < T2; i++) {
      for (int j = 0; j < T3; j++) {
        mat2[i * T3 + j] = i * T3 + j + 1;
      }
    }

    printf("mat2:\n");
    print_mat(mat2, T2, T3);

    int* result = new int[T1 * T3];
    // set result to 0
    for (int i = 0; i < T1; i++) {
      for (int j = 0; j < T3; j++) {
        result[i * T3 + j] = 0;
      }
    }

    mul_mat_mat<int, T1, T2, T3>(mat1, mat2, result);

    printf("result:\n");
    print_mat(result, T1, T3);
    // should be
    // 13 16
    // 40 52
    // 67 88
    // 94 124
  }
  return 0;
}