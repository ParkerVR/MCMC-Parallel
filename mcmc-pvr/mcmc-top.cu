// Top level program to be run

// to compile:
// 

// to compile with gcc if only c code is used:
// gcc -x c mcmc.cu 

#include "./markov-config.cuh"
#include "./markov-tools.cuh"

#if ENABLE_SERIAL
  #include "mcmc-serial.cuh"
#endif

#if ENABLE_GPU
  #include "mcmc-parallel.cuh"
#endif

int main(){
  float* arr;
  arr = malloc(sizeof(num_t) * ARR_TOT);
  return 0;
}