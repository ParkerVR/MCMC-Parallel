// MCMC Serial Algorithm


i_t checkFinished(num_t* arr, i_t lg, i_t row);
i_t nextRow(num_t* arr, i_t lg, i_t row);
i_t mcmc_serial(num_t* arr, i_t lg);

i_t checkFinished(num_t* arr, i_t lg, i_t row) {
  if (arr[row*lg+row] == 1.0)
    return 1;
  else return 0;
}

// 'Statistical' Search for closest value <= to random
i_t nextRow(num_t* arr, i_t lg, i_t row) {
  
  num_t random;
  random = (num_t)rand()/(num_t)(RAND_MAX);
  
  i_t index;
 
  index = lg*random;

  while (index != lg-1 && random > arr[lg*row + index])
    index++;

  while (index != 0 && random < arr[lg*row + index-1])
    index--;

#if PRINT_STEPS  
  printf("\n Current Row: %I64d", row);
  printf("\n Random: %1.3f", random);
  printf("\n Next Row: %I64d\n", index);
#endif

  return index;

}

i_t mcmc_serial(num_t* arr, i_t lg) {
  i_t index = 0;
  i_t step_count = 0;
  while( !checkFinished(arr, lg, index) ){
    index = nextRow(arr, lg, index);
    step_count++;
  }
  printf("\nSerial complete in %I64d steps.", step_count);
  return index;
}