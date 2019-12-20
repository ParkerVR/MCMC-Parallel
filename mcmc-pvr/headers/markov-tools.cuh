// Markov Chain Monte Carlo Helper Tools & Sys Includes

// gcc -x c mcmc.cu -o mcmc_s
#if COMPILER == GCC
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>


// g++ -x c mcmc.cu -o mcmc_s
#elif COMPILER == GPP
  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>
  
//
#elif COMPILER == CUDA
  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>

#else
  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>

#endif



#if ENABLE_GPU
  #include "cuPrintf.cu"
  #include "cuPrintf.cuh"
#endif

void arr_print(num_t* arr, int lg);
void arr_zero(num_t* arr, int sz);
void arr_init_cum_rand(num_t*arr, int lg, int endstates, int seed);


void array_init_cum_rand(num_t *arr, int lg, int endstates, int seed) {
  arr_zero(arr, lg);
  int i, j;
  num_t randNum;
  srand(seed);

  // Generate cumulative random rows that end at 1
  num_t rowSum;
  for (i = 0; i < lg-endstates; i++) {
    rowSum = 0;
    for (j = 0; j < lg; j++) {
      randNum = (num_t) rand();
      rowSum += randNum;
      arr[i*lg + j] = rowSum;
      //printf("%f\n", arr[i*lg+j]);
    }
    for (j = 0; j < lg; j++) {
      arr[i*lg + j] = arr[i*lg + j] / rowSum;
    }
  }

  // Define endstates arr[i,i] = 1
  // Note based on randomization process currently, last row will always be an end state
  for(i = lg - endstates; i < lg; i++) {
    arr[i*lg + i] = 1.0;
  }

  //printarr(arr,lg);
}

void arr_zero(num_t* arr, int lg) {
  int i, j;
  for(i = 0; i < lg; i++)
    for(j = 0; j < lg; j++)
      arr[i*lg+j] = (num_t)0.0;
}

void arr_print(num_t* arr, int lg) {
  int i, j;
  
  char printstr[] = "%1.2f"; //default print 4 chars
#if PRINT_DECIMALS < 0 || PRINT_DECIMALS > 9
  #warning PRINT_DECIMALS unsupported, using default (2)
#elif PRINT_DECIMALS
  printstr[3] = '0' + PRINT_DECIMALS; 
#endif
  printf("\n");
  for(i = 0; i < lg; i++){
    for(j = 0; j < lg; j++) {
      printf("%1.3f ", arr[i*lg+j] );
    }
    printf("\n");
  }

}