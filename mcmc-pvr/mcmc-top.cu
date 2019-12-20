// Top level program to be run

// Go to config or readme for compile instuctions


#include "mcmc-config.cuh" // The config can be swapped
#include "mcmc-headers.cuh"



#if ENABLE_SERIAL
  #include "mcmc-serial.cuh"
#endif

#if ENABLE_GPU
  #include "mcmc-parallel.cuh"
#endif

int main(){

  int sz = ARR_SZ;
  int lg = ARR_LG;

 

  num_t* arr;
  arr = malloc(sizeof(num_t) * sz);

  int seed = 420;
  int endstates = ENDSTATES;
  arr_init_cum_rand(arr, lg, endstates, seed);


  arr_print(arr, lg);

  return 0;
}



// printf("%d", PTRDIFF_MAX); // Used to determine practical max bytes per array

// MANUAL TESTING ARRAY
/* 
num_t arr[ARR_SZ]= {
  (num_t) 0.0, (num_t) 0.3, (num_t) 0.6, (num_t) 1.0,
  (num_t) 0.1, (num_t) 0.2, (num_t) 0.3, (num_t) 1.0,
  (num_t) 0.5, (num_t) 0.6, (num_t) 0.7, (num_t) 1.0,
  (num_t) 0.0, (num_t) 0.0, (num_t) 0.0, (num_t) 1.0,
};
*/