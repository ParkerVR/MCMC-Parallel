// MCMC Array Tools

// Print an lg*lg array to console
void arr_print(num_t* arr, int lg);

// Zero an array
void arr_zero(num_t* arr, int sz);

// Initialize lg*lg array with cumulative rows of randoms and variable number of endstates
void arr_init_cum_rand(num_t*arr, int lg, int endstates, int seed);



void arr_init_cum_rand(num_t *arr, int lg, int endstates, int seed) {
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
#if PRINT_N_DECIMALS < 0 || PRINT_N_DECIMALS > 9
  #warning "PRINT_N_DECIMALS unsupported, using default (2)"
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