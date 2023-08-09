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
  double yy[SEQ_LEN][8] = {0};
  // for every element in the sequence, apply front linear
  // front_linear is 8x2
  // xx is 8x2
  // so yy is 8x1
  for (int i = 0; i < SEQ_LEN; i++) {
    matmul<double, 8, 2, 1>(FRONT_LINEAR_WEIGHT, xx[i], yy[i]);
    add_bias<double, 8>(yy[i], FRONT_LINEAR_BIAS);
  }
  cout << "yy: " << endl;
  print_mat_template<double, SEQ_LEN, 8>(yy);

  // qkv proj
  double qkv[SEQ_LEN][3 * 8] = {0};
  for (int i = 0; i < SEQ_LEN; i++) {
    matmul<double, 3 * 8, 8, 1>(
        TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_QKV_PROJ_WEIGHT, yy[i], qkv[i]);
  }

  cout << "qkv: " << endl;
  print_mat_template<double, SEQ_LEN, 3 * 8>(qkv);

  // separate query key and value

  // reshape qkv
  double qkv_reshaped[SEQ_LEN][NUM_HEADS][3 * HEAD_DIM] = {0};
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int h = 0; h < NUM_HEADS; ++h) {
      for (int j = 0; j < 3 * HEAD_DIM; ++j) {
        qkv_reshaped[s][h][j] = qkv[s][h * (3 * 4) + j];
      }
    }
  }
  cout << "qkv_reshaped: " << endl;
  for (int s = 0; s < SEQ_LEN; ++s) {
    cout << "seq " << s << endl;
    print_mat_template<double, NUM_HEADS, 3 * HEAD_DIM>(qkv_reshaped[s]);
  }

  // transpose NUM_HEADS and SEQ_LEN
  double qkv_reshaped_transposed[NUM_HEADS][SEQ_LEN][3 * HEAD_DIM] = {0};
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int h = 0; h < NUM_HEADS; ++h) {
      for (int j = 0; j < 3 * HEAD_DIM; ++j) {
        qkv_reshaped_transposed[h][s][j] = qkv_reshaped[s][h][j];
      }
    }
  }

  cout << "qkv_reshaped_transposed: " << endl;
  for (int h = 0; h < NUM_HEADS; ++h) {
    cout << "head " << h << endl;
    print_mat_template<double, SEQ_LEN, 3 * HEAD_DIM>(
        qkv_reshaped_transposed[h]);
  }

  // split into query key and value
  double q[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};
  double k[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};
  double v[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};

  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int s = 0; s < SEQ_LEN; ++s) {
      for (int j = 0; j < HEAD_DIM; ++j) {
        q[h][s][j] = qkv_reshaped_transposed[h][s][j];
        k[h][s][j] = qkv_reshaped_transposed[h][s][j + HEAD_DIM];
        v[h][s][j] = qkv_reshaped_transposed[h][s][j + 2 * HEAD_DIM];
      }
    }
  }

  for (int h = 0; h < NUM_HEADS; ++h) {
    cout << "head " << h << endl;
    cout << "q: " << endl;
    print_mat_template<double, SEQ_LEN, HEAD_DIM>(q[h]);
    cout << "k: " << endl;
    print_mat_template<double, SEQ_LEN, HEAD_DIM>(k[h]);
    cout << "v: " << endl;
    print_mat_template<double, SEQ_LEN, HEAD_DIM>(v[h]);
    cout << endl;
  }

  // calculate self-attention
  double k_transposed[NUM_HEADS][HEAD_DIM][SEQ_LEN] = {0};
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int j = 0; j < HEAD_DIM; ++j) {
      for (int s = 0; s < SEQ_LEN; ++s) {
        k_transposed[h][j][s] = k[h][s][j];
      }
    }
  }
  double attention_logits[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {0};
  for (int h = 0; h < NUM_HEADS; ++h) {
    matmul<double, SEQ_LEN, HEAD_DIM, SEQ_LEN>(q[h][0], k_transposed[h][0],
                                               attention_logits[h][0]);
    // normalize
    for (int s = 0; s < SEQ_LEN; ++s) {
      for (int t = 0; t < SEQ_LEN; ++t) {
        attention_logits[h][s][t] /= sqrt(4);
      }
    }
  }

  cout << "attention_logits: " << endl;
  for (int h = 0; h < NUM_HEADS; ++h) {
    cout << "head " << h << endl;
    print_mat_template<double, SEQ_LEN, SEQ_LEN>(attention_logits[h]);
  }

  // softmax
  // subtract max from attention_logits numerical stability
  double max[NUM_HEADS][SEQ_LEN] = {0};
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int s = 0; s < SEQ_LEN; ++s) {
      max[h][s] = attention_logits[h][s][0];
      for (int t = 1; t < SEQ_LEN; ++t) {
        if (attention_logits[h][s][t] > max[h][s]) {
          max[h][s] = attention_logits[h][s][t];
        }
      }
    }
  }

  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int s = 0; s < SEQ_LEN; ++s) {
      for (int t = 0; t < SEQ_LEN; ++t) {
        attention_logits[h][s][t] -= max[h][s];
      }
    }
  }
  cout << "attention_logits after subtract max: " << endl;
  for (int h = 0; h < NUM_HEADS; ++h) {
    cout << "head " << h << endl;
    print_mat_template<double, SEQ_LEN, SEQ_LEN>(attention_logits[h]);
  }

  double attention_weights[NUM_HEADS][SEQ_LEN][SEQ_LEN] = {0};
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int s = 0; s < SEQ_LEN; ++s) {
      softmax<double, SEQ_LEN>(attention_logits[h][s], attention_weights[h][s]);
    }
  }

  cout << "attention_weights: " << endl;
  for (int h = 0; h < NUM_HEADS; ++h) {
    cout << "head " << h << endl;
    print_mat_template<double, SEQ_LEN, SEQ_LEN>(attention_weights[h]);
  }

  // compute the values
  double attention_output[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int s = 0; s < SEQ_LEN; ++s) {
      for (int j = 0; j < HEAD_DIM; ++j) {
        for (int t = 0; t < SEQ_LEN; ++t) {
          attention_output[h][s][j] += attention_weights[h][s][t] * v[h][t][j];
        }
      }
    }
  }

  cout << "attention_output: " << endl;
  for (int h = 0; h < NUM_HEADS; ++h) {
    cout << "head " << h << endl;
    print_mat_template<double, SEQ_LEN, HEAD_DIM>(attention_output[h]);
  }

  // permute the heads and seq len
  double values_transposed[SEQ_LEN][NUM_HEADS][HEAD_DIM] = {0};
  for (int h = 0; h < NUM_HEADS; ++h) {
    for (int j = 0; j < HEAD_DIM; ++j) {
      for (int s = 0; s < SEQ_LEN; ++s) {
        values_transposed[s][h][j] = attention_output[h][s][j];
      }
    }
  }

  cout << "values_transposed: " << endl;
  for (int s = 0; s < SEQ_LEN; ++s) {
    cout << "seq " << s << endl;
    print_mat_template<double, NUM_HEADS, HEAD_DIM>(values_transposed[s]);
  }

  // reshape into seq_len x (num_heads * head_dim)
  double values_reshaped[SEQ_LEN][NUM_HEADS * HEAD_DIM] = {0};
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int h = 0; h < NUM_HEADS; ++h) {
      for (int j = 0; j < HEAD_DIM; ++j) {
        values_reshaped[s][h * HEAD_DIM + j] = values_transposed[s][h][j];
      }
    }
  }

  cout << "values_reshaped: " << endl;
  print_mat_template<double, SEQ_LEN, NUM_HEADS * HEAD_DIM>(values_reshaped);

  // project
  double values_proj[SEQ_LEN][8] = {0};
  matmul<double, SEQ_LEN, NUM_HEADS * HEAD_DIM, 8>(
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_WEIGHT,
      (const double *)values_reshaped, (double *)values_proj);
  // add bias
  for (int s = 0; s < SEQ_LEN; ++s) {
    add_bias<double, 8>((double *)values_proj[s],
                        TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_BIAS);
  }

  cout << "values_proj: " << endl;
  print_mat_template<double, SEQ_LEN, 8>(values_proj);

  return 0;
}