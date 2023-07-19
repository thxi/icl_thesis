#include <stdio.h>
#include <string.h>

void toplevel(volatile int *a, int length, int value){
#pragma HLS INTERFACE m_axi port=a depth=50 offset=slave
#pragma HLS INTERFACE s_axilite port=length
#pragma HLS INTERFACE s_axilite port=value
#pragma HLS INTERFACE s_axilite port=return

	int i;
  int buff[100];

  //memcpy creates a burst access to memory
  //multiple calls of memcpy cannot be pipelined and will be scheduled sequentially
  //memcpy requires a local buffer to store the results of the memory transaction
  memcpy(buff,(const int*)a,length*sizeof(int));

  for(i=0; i < length; i++){
    buff[i] = buff[i] + value;
  }

  memcpy((int *)a,buff,length*sizeof(int));
}
