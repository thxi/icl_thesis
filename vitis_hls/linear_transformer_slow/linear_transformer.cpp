#include "linear_transformer.h"

#include <stdio.h>

#include <iostream>

#include "scaled_dp_params.h"

#define EPS 0.000001  // for normalizer numerical stability

void print_mat(const dout_t* mat, int T1, int T2) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      printf("%.6f ", mat[i * T2 + j]);
    }
    printf("\n");
  }
}

using namespace std;

// input_array should be of size SEQ_LEN x INPUT_DIM
void transformer(din_t* input_array, dout_t* val) {
#pragma HLS INTERFACE m_axi port = input_array depth = 16 offset = \
    slave bundle = axi_ports
#pragma HLS INTERFACE m_axi port = val depth = 1 offset = slave bundle = \
    axi_ports
#pragma HLS INTERFACE s_axilite port = return

  din_t xx[SEQ_LEN][INPUT_DIM] = {0};
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < INPUT_DIM; j++) {
      xx[i][j] = input_array[i * INPUT_DIM + j];
    }
  }

  val[0] = transformer_for_sample(xx);
}

dout_t transformer_for_sample(din_t xx[SEQ_LEN][INPUT_DIM]) {
// for every element in the sequence, apply front linear
// front_linear is 8x2 but it is stored as a transposed matrix
// so multiply on the right
// xx is 8x2
// so yy is 8x8
front_linear:
  dout_t front_linear_output[SEQ_LEN][BLOCK_INPUT_DIM] = {0};

  kernel_mmult<dout_t, SEQ_LEN, INPUT_DIM, BLOCK_INPUT_DIM>(
      xx, FRONT_LINEAR_WEIGHT, front_linear_output);
  kernel_bias_add<dout_t, SEQ_LEN, BLOCK_INPUT_DIM>(
      front_linear_output, FRONT_LINEAR_BIAS, front_linear_output);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "front_linear_output: " << endl;
  print_mat((dout_t*)front_linear_output, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  // Transformer encoder
  // calculate query key and value matrices
  dout_t query_mat[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  dout_t key_mat[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  dout_t value_mat[SEQ_LEN][BLOCK_INPUT_DIM] = {0};

query_mat:
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      front_linear_output,
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_QUERY_MAT_WEIGHT, query_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "query_mat: " << endl;
  print_mat((dout_t*)query_mat, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

key_mat:
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      front_linear_output,
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_KEY_MAT_WEIGHT, key_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "key_mat: " << endl;
  print_mat((dout_t*)key_mat, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

value_mat:
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      front_linear_output,
      TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_VALUE_MAT_WEIGHT, value_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "value_mat: " << endl;
  print_mat((dout_t*)value_mat, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  elu_inplace<dout_t, SEQ_LEN, BLOCK_INPUT_DIM>(query_mat);
  elu_inplace<dout_t, SEQ_LEN, BLOCK_INPUT_DIM>(key_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "query_mat after elu: " << endl;
  print_mat((dout_t*)query_mat, SEQ_LEN, BLOCK_INPUT_DIM);
  cout << "key_mat after elu: " << endl;
  print_mat((dout_t*)key_mat, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  // transpose value

  dout_t value_mat_transposed[BLOCK_INPUT_DIM][SEQ_LEN] = {0};
transpose_value_mat:
  for (int i = 0; i < SEQ_LEN; i++) {
#pragma HLS PIPELINE off
    for (int j = 0; j < BLOCK_INPUT_DIM; j++) {
#pragma HLS PIPELINE off
      value_mat_transposed[j][i] = value_mat[i][j];
    }
  }

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "value_mat_transposed: " << endl;
  print_mat((dout_t*)value_mat_transposed, BLOCK_INPUT_DIM, SEQ_LEN);
#endif

  // calculate kv
  dout_t kv_mat[SEQ_LEN][SEQ_LEN] = {0};
  // need to manually partition key_mat
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, SEQ_LEN>(
      key_mat, value_mat_transposed, kv_mat);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "kv_mat: " << endl;
  print_mat((dout_t*)kv_mat, SEQ_LEN, SEQ_LEN);
#endif

  // sum k along columns
  dout_t sum_k[SEQ_LEN][1] = {0};
sum_k:
  for (int j = 0; j < SEQ_LEN; j++) {
#pragma HLS PIPELINE off
  sum_k_inner:
    for (int i = 0; i < SEQ_LEN; i++) {
#pragma HLS PIPELINE off
      sum_k[j][0] += key_mat[i][j];
    }
  }

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "sum_k: " << endl;
  print_mat((dout_t*)sum_k, SEQ_LEN, 1);
#endif

  // transpose q for normalizer
  dout_t query_mat_transposed[BLOCK_INPUT_DIM][SEQ_LEN] = {0};
transpose_query_mat:
  for (int i = 0; i < SEQ_LEN; i++) {
#pragma HLS PIPELINE off
    for (int j = 0; j < BLOCK_INPUT_DIM; j++) {
#pragma HLS PIPELINE off
      query_mat_transposed[j][i] = query_mat[i][j];
    }
  }

  // compute normalizer Z
  dout_t normalizer[SEQ_LEN][1] = {0};

  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, 1>(query_mat_transposed, sum_k,
                                                    normalizer);

// inverse, so that don't have to do it when computing values_new multiple times
normalizer_inverse:
  for (int i = 0; i < SEQ_LEN; i++) {
#pragma HLS PIPELINE off
    normalizer[i][0] = 1 / (normalizer[i][0] + EPS);
  }

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "normalizer: " << endl;
  print_mat((dout_t*)normalizer, SEQ_LEN, 1);
#endif

  // numerator
  dout_t numerator[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
#pragma HLS ARRAY_PARTITION variable = query_mat complete dim = 2
  kernel_mmult<dout_t, SEQ_LEN, SEQ_LEN, BLOCK_INPUT_DIM>(query_mat, kv_mat,
                                                          numerator);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "numerator: " << endl;
  print_mat((dout_t*)numerator, SEQ_LEN, SEQ_LEN);
#endif

  // hadamart product (i.e., element-wise product) of numerator and normalizer
  // multiply each row of numerator by a single value in normalizer for the
  // corresponding row
  dout_t values_new[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
hadamart_loop_i:
  for (int i = 0; i < SEQ_LEN; i++) {
#pragma HLS PIPELINE off
  hadamart_loop_j:
    for (int j = 0; j < BLOCK_INPUT_DIM; j++) {
#pragma HLS PIPELINE off
      values_new[i][j] = numerator[i][j] * normalizer[i][0];
    }
  }

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "values_new: " << endl;
  print_mat((dout_t*)values_new, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  // o_proj, exiting the linear attention
  dout_t o_proj[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  // TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_WEIGHT
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      values_new, TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_WEIGHT, o_proj);
  kernel_bias_add<dout_t, SEQ_LEN, BLOCK_INPUT_DIM>(
      o_proj, TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_BIAS, o_proj);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "o_proj: " << endl;
  print_mat((dout_t*)o_proj, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  // linear net
  // TODO: maybe its a bit dirty that I hardcode the
  // 2 * BLOCK_INPUT_DIM here
  dout_t first_linear_net_output[SEQ_LEN][2 * BLOCK_INPUT_DIM] = {0};
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, 2 * BLOCK_INPUT_DIM>(
      o_proj, TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_0_WEIGHT,
      first_linear_net_output);
  kernel_bias_add<dout_t, SEQ_LEN, 2 * BLOCK_INPUT_DIM>(
      first_linear_net_output, TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_0_BIAS,
      first_linear_net_output);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "first_linear_net_output: " << endl;
  print_mat((dout_t*)first_linear_net_output, SEQ_LEN, 2 * BLOCK_INPUT_DIM);
#endif

  // ReLU
  relu_inplace<dout_t, SEQ_LEN, 2 * BLOCK_INPUT_DIM>(first_linear_net_output);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "first_linear_net_output after ReLU: " << endl;
  print_mat((dout_t*)first_linear_net_output, SEQ_LEN, 2 * BLOCK_INPUT_DIM);
#endif

  // second linear net
  dout_t second_linear_net_output[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  kernel_mmult<dout_t, SEQ_LEN, 2 * BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      first_linear_net_output, TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_3_WEIGHT,
      second_linear_net_output);
  kernel_bias_add<dout_t, SEQ_LEN, BLOCK_INPUT_DIM>(
      second_linear_net_output, TRANSFORMER_ENCODER_LAYERS_0_LINEAR_NET_3_BIAS,
      second_linear_net_output);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "second_linear_net_output: " << endl;
  print_mat((dout_t*)second_linear_net_output, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  // final_layer
  dout_t final_layer_output[SEQ_LEN][1] = {0};
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, 1>(
      second_linear_net_output, FINAL_LINEAR_WEIGHT, final_layer_output);
  kernel_bias_add<dout_t, SEQ_LEN, 1>(final_layer_output, FINAL_LINEAR_BIAS,
                                      final_layer_output);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "final_layer_output: " << endl;
  print_mat((dout_t*)final_layer_output, SEQ_LEN, 1);
#endif

  return final_layer_output[0][0];
}