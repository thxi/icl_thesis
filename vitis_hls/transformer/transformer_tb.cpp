#include <algorithm>
#include <iostream>

#include "transformer.h"

int main() {
  din_t xx[SEQ_LEN][INPUT_DIM] = {
      {0.2, 0.2}, {0.2, 0.2}, {0.4, 0.4}, {0.4, 0.4},
      {0.6, 0.6}, {0.6, 0.6}, {0.8, 0.8}, {0.8, 0.8},
  };

  din_t xx_flat[SEQ_LEN * INPUT_DIM];
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < INPUT_DIM; j++) {
      xx_flat[i * INPUT_DIM + j] = xx[i][j];
    }
  }

  dout_t tf_output = -1234567;
  transformer(xx_flat, tf_output);
  std::cout << "tf_output: " << tf_output << std::endl;
}
