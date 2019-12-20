// Speculative Execution tools


// Creates stage speculative execution table
void get_spec_table(num_t* arr, num_t* table, i_t lg, i_t endpoints);
void print_spec_table(num_t* table, i_t lg);


void print_spec_table(num_t* table, i_t lg) {

  i_t i;
  printf("\nSpeculative Execution Table\n");
  for(i = 0; i < lg; i++){
    printf("  %I64d | %I64d\n",i,table[i]);
  }

}

void get_spec_table(num_t* arr, num_t* table, i_t lg, i_t endpoints) {

  // This can be made more efficient by sampling random rows -- works better on large table
  // For our purposes the probability will average out for each operation - need a different randomizer for true effect

  i_t i, j;
  num_t sum = 0;

  // If we lose too much accuracy, we can always use two arrays and divide by sum each row

  i_t l = lg - endpoints;

  for(i = 0; i < l; i++) {
    for(j = 0; j < lg; j++) {
      table[j] += arr[i*lg + j];
      sum += arr[i*lg + j];
    }
  }

  for(i = 0; i < lg; i++) {
    table[i] = table[i] / sum;
  }

  /*
  // Make cumulative ?
  for(i = 1; i < l; i++) {
    table[i] += table[i-1];
  }
  */

}