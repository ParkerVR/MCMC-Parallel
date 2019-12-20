// Markov Chain Monte Carlo Helper Tools & Sys Includes

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

#if ENABLE_GPU
  #include "cuPrintf.cu"
  #include "cuPrintf.cuh"
#endif

void printarr(float* arr, int sz);
void initializeArray1D(float *arr, int sz, int seed);


void initializeArray1D(float *arr, int sz, int seed) {
  int i, j;
  float randNum;
  srand(seed);

  // Generate random rows that sum to probability of 1

  float rowSum;
  for (i = 0; i < sz-2; i++) {
    rowSum = 0;
    for (j = 0; j < sz; j++) {
      randNum = (float) rand();
      rowSum += randNum;
      arr[i*sz + j] = rowSum;
      printf("%f\n", arr[i*sz+j]);
    }
    for (j = 0; j < sz; j++) {
      arr[i*sz + j] = arr[i*sz + j] / rowSum;
    }
  }

  // Define (2) end states
  // Note based on rules currently, last number will always be end state
  for (int j = 0; j < sz; j++) {
    arr[(sz-2)*sz + j] = 0;
    arr[(sz-1)*sz + j] = 0;
  }

  arr[(sz-2)*sz + sz-2] = 1.0;
  arr[(sz-1)*sz + sz-1] = 1.0;

  printarr(arr,sz);
}


void printarr(float* arr, int sz) {
  int i, j;
  for(i = 0; i < sz; i++){
    for(j = 0; j < sz; j++) {
      printf("%f, ", arr[i*sz+j] );
    }
    printf("\n");
  }
}