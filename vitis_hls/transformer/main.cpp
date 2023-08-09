#include <algorithm>
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

  // transpose again
  double values_transposed2[NUM_HEADS * HEAD_DIM][SEQ_LEN] = {0};
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < NUM_HEADS * HEAD_DIM; ++j) {
      values_transposed2[j][s] = values_reshaped[s][j];
    }
  }

  // project
  double values_proj[SEQ_LEN][8] = {0};
  matmul<double, SEQ_LEN, NUM_HEADS * HEAD_DIM, 8>(
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_WEIGHT,
      (const double *)values_transposed2, (double *)values_proj);
  // transpose
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < s; ++j) {
      double tmp = values_proj[s][j];
      values_proj[s][j] = values_proj[j][s];
      values_proj[j][s] = tmp;
    }
  }
  // add bias
  for (int s = 0; s < SEQ_LEN; ++s) {
    add_bias<double, 8>((double *)values_proj[s],
                        TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_BIAS);
  }

  cout << "values_proj: " << endl;
  print_mat_template<double, SEQ_LEN, 8>(values_proj);

  // add residual
  // TODO: maybe reuse yy array
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < 8; ++j) {
      values_proj[s][j] += yy[s][j];
    }
  }

  cout << "values_proj + residual: " << endl;
  print_mat_template<double, SEQ_LEN, 8>(values_proj);

  // transpose again
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < s; ++j) {
      double tmp = values_proj[s][j];
      values_proj[s][j] = values_proj[j][s];
      values_proj[j][s] = tmp;
    }
  }

  cout << "values_proj + residual transposed: " << endl;
  print_mat_template<double, SEQ_LEN, 8>(values_proj);

  // MLP feed forward
  double ff1[16][8] = {0};
  matmul<double, 16, SEQ_LEN, 8>(
      TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_0_WEIGHT,
      (const double *)values_proj, (double *)ff1);

  for (int o_dim = 0; o_dim < 16; ++o_dim) {
    for (int s = 0; s < 8; ++s) {
      ff1[o_dim][s] += TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_0_BIAS[o_dim];
    }
  }

  // transpose + relu
  double ff1_transposed[8][16] = {0};
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < 16; ++j) {
      ff1_transposed[s][j] = std::max(ff1[j][s], 0.0);
    }
  }

  cout << "ff1_transposed: " << endl;
  print_mat_template<double, 8, 16>(ff1_transposed);

  // ff2
  double ff2[8][8] = {0};

  matmul<double, 8, 16, 8>(TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_3_WEIGHT,
                           (const double *)ff1, (double *)ff2);

  for (int o_dim = 0; o_dim < 8; ++o_dim) {
    for (int s = 0; s < 8; ++s) {
      ff2[o_dim][s] += TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_3_BIAS[o_dim];
    }
  }

  // transpose
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < s; ++j) {
      double tmp = ff2[s][j];
      ff2[s][j] = ff2[j][s];
      ff2[j][s] = tmp;
    }
  }

  cout << "ff2_transposed: " << endl;
  print_mat_template<double, 8, 8>(ff2);

  // transpose o_proj + residual
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < s; ++j) {
      double tmp = values_proj[s][j];
      values_proj[s][j] = values_proj[j][s];
      values_proj[j][s] = tmp;
    }
  }

  // add residual
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < 8; ++j) {
      ff2[s][j] += values_proj[s][j];
    }
  }

  cout << "ff2_transposed + residual: " << endl;
  print_mat_template<double, 8, 8>(ff2);

  // transpose again
  for (int s = 0; s < SEQ_LEN; ++s) {
    for (int j = 0; j < s; ++j) {
      double tmp = ff2[s][j];
      ff2[s][j] = ff2[j][s];
      ff2[j][s] = tmp;
    }
  }

  // calculate output, only need 1 value
  double out = FINAL_LINEAR_BIAS[0];
  for (int s = 0; s < SEQ_LEN; ++s) {
    out += FINAL_LINEAR_WEIGHT[s] * ff2[s][0];
  }

  cout << "out: " << out << endl;

  return 0;
}