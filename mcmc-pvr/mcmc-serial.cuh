// MCMC Serial Algorithm


int checkFinished(num_t* arr, int lg, int row);
int nextRow(num_t* arr, int lg, int row);
int mcmc_serial(num_t* arr, int lg);

int checkFinished(num_t* arr, int lg, int row) {
  if (arr[row*lg+row] == 1.0)
    return 1;
  else return 0;
}

// 'Statistical' Search for closest value <= to random
int nextRow(num_t* arr, int lg, int row) {
  
  num_t random;
  random = (num_t)rand()/(num_t)(RAND_MAX);
  
  int index;
 
  index = lg*random;

  while (index != lg-1 && random > arr[lg*row + index])
    index++;

  while (index != 0 && random < arr[lg*row + index-1])
    index--;

#if PRINT_STEPS  
  printf("\n Current Row: %d", row);
  printf("\n Random: %1.3f", random);
  printf("\n Next Row: %d\n", index);
#endif

  return index;

}

int mcmc_serial(num_t* arr, int lg) {
  int index = 0;
  while( !checkFinished(arr, lg, index) ){
    index = nextRow(arr, lg, index);
  }
  return index;
}