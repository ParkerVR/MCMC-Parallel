// MARKOV CHAIN MONTE CARLO MAIN FILE

// Go to config or readme for compile instuctions


#include "mcmc-config.cuh" // The config can be swapped
#include "mcmc-headers.cuh" // Lists all internal includes




int main() {

  
  i_t sz = ARR_SZ;
  i_t lg = ARR_LG; 

  num_t* arr;
  arr = malloc(sizeof(num_t) * sz);


  #if SEEDED
    int seed = SEED;
  #else
    int seed = time(0);
    printf("\nSeed Generated: %d", seed);
  #endif

  i_t endstates = ENDSTATES;
  arr_init_cum_rand(arr, lg, endstates, seed);

  #if PRINT_ARR
    arr_print(arr, lg);
  #endif

  
  #if ENABLE_GPU
  
    time_g gpu_timer = gpu_timer_start();



    #if PRINT_TIME
      float gpu_time = gpu_time_elapsed(gpu_timer);
      printf("\nGPU Time Elapsed: %1.3f\n", gpu_time);
    #endif

  #endif
  

  #if ENABLE_SERIAL

    clock_t cpu_timer = cpu_timer_start();

    i_t outRow = mcmc_serial(arr, lg) - lg + endstates + 1;

    #if PRINT_RESULT
      printf("\n LG = %ld FINISHED AT OUTROW # %ld", lg, outRow);
    #endif

    #if PRINT_TIME
      float cpu_time = cpu_time_elapsed(cpu_timer);
      printf("\nCPU Time Elapsed: %1.3f\n", cpu_time);
    #endif

  #endif

  return 0;
}


// printf("%lu", sizeof(i_t)); // Tells num bytes in data type
// printf("%d", PTRDIFF_MAX); // Used to determine practical max bytes per array

