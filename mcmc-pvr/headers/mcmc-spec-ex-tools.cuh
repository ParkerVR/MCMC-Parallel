// Speculative Execution tools


// Creates stage speculative execution table
void get_spec_table(num_t* arr, num_t* table, i_t l);
void print_spec_table(num_t* table, i_t lg, i_t l);


void print_spec_table(num_t* table, i_t l) {

  i_t i;
  printf("\nSpeculative Execution Table\n");
  for(i = 0; i < l; i++) {
    printf("  %I64d | %f\n",i,table[i]);
  }

}

void get_spec_table(num_t* arr, num_t* table, i_t lg, i_t l) {

  // This can be made more efficient by sampling random rows -- works better on large table
  // For our purposes the probability will average out for each operation - need a different randomizer for true effect

  i_t i, j;
  num_t sum = 0;

  // If we lose too much accuracy, we can always use two arrays and divide by sum each row

  for(i = 0; i < l; i++) {

    table[0] += arr[i*lg];
    sum += arr[i*lg];

    for(j = 1; j < l; j++) {
      table[j] += arr[i*lg + j] - arr[i*lg + j-1];
      sum += arr[i*lg + j] - arr[i*lg + j-1];
    }
  }

  for(i = 0; i < l; i++) {
    table[i] = table[i] / sum;
  }

  
  // Make cumulative
  for(i = 1; i < l; i++) {
    table[i] += table[i-1];
  }
  
  // Deal with roundoff
  table[l-1] = 1.0;

}