#include <algorithm>
#include <iostream>

#include "transformer.h"

int main() {
  din_t xx[SEQ_LEN][INPUT_DIM] = {
      {0.2, 0.2}, {0.2, 0.2}, {0.4, 0.4}, {0.4, 0.4},
      {0.6, 0.6}, {0.6, 0.6}, {0.8, 0.8}, {0.8, 0.8},
  };

  dout_t tf_output = transformer(xx);
  std::cout << "tf_output: " << tf_output << std::endl;
}
