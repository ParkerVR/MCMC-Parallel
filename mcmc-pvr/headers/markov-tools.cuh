// Markov Chain Monte Carlo Helper Tools & Sys Includes

// gcc -x c mcmc.cu -o mcmc_s
#if COMPILER == GCC
  #include <stdlib.h>
  #include <stdio.h>
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

// g++ -x c mcmc.cu -o mcmc_s
#elif COMPILER == GPP
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

void printarr(num_t* arr, int lg);
void zeroarr(num_t* arr, int sz);
void initializeArray1D(num_t*arr, int lg, int endstates, int seed);


void initializeArray1D(num_t *arr, int lg, int endstates, int seed) {
  zeroarr(arr, lg);
  int i, j;
  num_t randNum;
  srand(seed);

  // Generate random rows that sum to probability of 1

  num_t rowSum;
  for (i = 0; i < lg-2; i++) {
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

  // Define (2) end states
  // Note based on rules currently, last number will always be end state
  for (j = 0; j < lg; j++) {
    arr[(lg-2)*lg + j] = 0;
    arr[(lg-1)*lg + j] = 0;
  }

  arr[(lg-2)*lg + lg-2] = 1.0;
  arr[(lg-1)*lg + lg-1] = 1.0;

  //printarr(arr,lg);
}

void zeroarr(num_t* arr, int lg) {
  int i, j;
  for(i = 0; i < lg; i++)
    for(j = 0; j < lg; j++)
      arr[i*lg+j] = (num_t)0.0;
}

void printarr(num_t* arr, int lg) {
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