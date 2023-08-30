#include "transformer.h"

#include <stdio.h>

#include <iostream>

#include "scaled_dp_params.h"

void print_mat(const dout_t* mat, int T1, int T2) {
  for (int i = 0; i < T1; i++) {
    for (int j = 0; j < T2; j++) {
      printf("%.6f ", mat[i * T2 + j]);
    }
    printf("\n");
  }
}

using namespace std;

dout_t transformer(din_t xx[SEQ_LEN][INPUT_DIM]) {
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

  // split into multiple heads
  dout_t query_mat_split[SEQ_LEN][NUM_HEADS][HEAD_DIM] = {0};
  dout_t key_mat_split[SEQ_LEN][NUM_HEADS][HEAD_DIM] = {0};
  dout_t value_mat_split[SEQ_LEN][NUM_HEADS][HEAD_DIM] = {0};

split_heads:
  split_head<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(query_mat, query_mat_split);
  split_head<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(key_mat, key_mat_split);
  split_head<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(value_mat, value_mat_split);

  dout_t query_mat_transposed[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};
  dout_t key_mat_transposed[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};
  dout_t value_mat_transposed[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};

transpose_heads:
  // transpose first 2 dimensions
  transpose_3d<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(query_mat_split,
                                                     query_mat_transposed);
  transpose_3d<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(key_mat_split,
                                                     key_mat_transposed);
  transpose_3d<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(value_mat_split,
                                                     value_mat_transposed);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "query_mat_transposed: " << endl;
  print_mat_3d<dout_t, NUM_HEADS, SEQ_LEN, HEAD_DIM>(query_mat_transposed);

  cout << "key_mat_transposed: " << endl;
  print_mat_3d<dout_t, NUM_HEADS, SEQ_LEN, HEAD_DIM>(key_mat_transposed);

  cout << "value_mat_transposed: " << endl;
  print_mat_3d<dout_t, NUM_HEADS, SEQ_LEN, HEAD_DIM>(value_mat_transposed);
#endif

  // calculate multihead attention
  dout_t scaled_dot_product_attention[NUM_HEADS][SEQ_LEN][HEAD_DIM] = {0};
  multihead_attention<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(
      query_mat_transposed, key_mat_transposed, value_mat_transposed,
      scaled_dot_product_attention);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "scaled_dot_product_attention: " << endl;
  print_mat_3d<dout_t, NUM_HEADS, SEQ_LEN, HEAD_DIM>(
      scaled_dot_product_attention);
#endif

  // transpose back
  dout_t scaled_dot_product_attention_transposed[SEQ_LEN][NUM_HEADS][HEAD_DIM] =
      {0};
  transpose_3d<dout_t, NUM_HEADS, SEQ_LEN, HEAD_DIM>(
      scaled_dot_product_attention, scaled_dot_product_attention_transposed);
  // so now it is SEQ_LEN x NUM_HEADS x HEAD_DIM

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "scaled_dot_product_attention_transposed: " << endl;
  print_mat_3d<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(
      scaled_dot_product_attention_transposed);
#endif

  // concat heads
  dout_t attention_values[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  concat_head<dout_t, SEQ_LEN, NUM_HEADS, HEAD_DIM>(
      scaled_dot_product_attention_transposed, attention_values);

#ifdef PRINT_INTERMEDIATE_RESULTS
  cout << "attention_values: " << endl;
  print_mat((dout_t*)attention_values, SEQ_LEN, BLOCK_INPUT_DIM);
#endif

  // o_proj, exiting the multihead attention
  dout_t o_proj[SEQ_LEN][BLOCK_INPUT_DIM] = {0};
  kernel_mmult<dout_t, SEQ_LEN, BLOCK_INPUT_DIM, BLOCK_INPUT_DIM>(
      attention_values, TRANSFORMER_ENCODER_LAYERS_0_SELF_ATTN_O_PROJ_WEIGHT,
      o_proj);
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