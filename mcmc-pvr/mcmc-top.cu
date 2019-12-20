// Top level program to be run

// Go to config or readme for compile instuctions

#include "./headers/markov-config.cuh"
#include "./headers/markov-tools.cuh"

#if ENABLE_SERIAL
  #include "mcmc-serial.cuh"
#endif

#if ENABLE_GPU
  #include "mcmc-parallel.cuh"
#endif

int main(){

  int sz = ARR_SZ;
  int lg = ARR_LG;
  // printf("%d", PTRDIFF_MAX); // Used to determine practical max bytes per array
  // ARRAY DEBUG GENERATOR
  /* 
  num_t arr[ARR_SZ]= {
    (num_t) 0.0, (num_t) 1.1, (num_t) 2.2, (num_t) 3.3,
    (num_t) 1.1, (num_t) 2.2, (num_t) 3.3, (num_t) 4.4,
    (num_t) 5.5, (num_t) 6.6, (num_t) 7.7, (num_t) 8.8,
    (num_t) 9.9, (num_t) 0.0, (num_t) 1.1, (num_t) 2.2
  };
  */

  num_t* arr;
  arr = malloc(sizeof(num_t) * sz);

  int seed = 420;
  int endstates = ENDSTATES;
  arr_init_cum_rand(arr, lg, endstates, seed);


  arr_print(arr, lg);

  return 0;
}