#ifndef __MATRIXMUL_H__
#define __MATRIXMUL_H__

#include <cmath>
using namespace std;

// Uncomment this line to compare TB vs HW C-model and/or RTL
// #define HW_COSIM

#define MAT_A_ROWS 3
#define MAT_A_COLS 4
#define MAT_B_ROWS 4
#define MAT_B_COLS 3

typedef int32_t mat_a_t;
typedef int32_t mat_b_t;
typedef int32_t result_t;

// Prototype of top level function for C-synthesis
void matrixmul(mat_a_t a[MAT_A_ROWS][MAT_A_COLS],
               mat_b_t b[MAT_B_ROWS][MAT_B_COLS],
               result_t res[MAT_A_ROWS][MAT_B_COLS]);

#endif  // __MATRIXMUL_H__ not defined
