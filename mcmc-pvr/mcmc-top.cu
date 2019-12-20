// MARKOV CHAIN MONTE CARLO MAIN FILE

// Go to config or readme for compile instuctions


#include "mcmc-config.cuh" // The config can be swapped
#include "mcmc-headers.cuh" // Lists all internal includes




int main() {

  #if PRINT_TIME
      clock_t setup_timer = cpu_timer_start();
  #endif
  i_t sz = ARR_SZ;
  i_t lg = ARR_LG; 

  num_t* arr;
  arr = (num_t*)malloc(sizeof(num_t) * sz);

  printf("\nTesting %I64d nodes with %d endstates.\n", lg, ENDSTATES);

  #if SEEDED
    int seed = SEED;
  #else
    int seed = time(0);
    printf("\nSeed Generated: %d\n", seed);
  #endif

  i_t endstates = ENDSTATES;
  arr_init_cum_rand(arr, lg, endstates, seed);

  #if PRINT_TIME
    float setup_time = cpu_time_elapsed(setup_timer);
    printf("\nSetup Time Elapsed: %1.3f seconds\n", setup_time);
  #endif

  #if PRINT_ARR
    arr_print(arr, lg);
  #endif

  
  #if ENABLE_GPU
    i_t spec_lg = lg - endstates;
    num_t* spec_table = (num_t*)malloc(sizeof(num_t) * (spec_lg));
    get_spec_table(arr, spec_table, lg, spec_lg);

    #if PRINT_SPEC
      print_spec_table(spec_table, spec_lg);
    #endif

    #if PRINT_TIME
      time_g gpu_timer = gpu_timer_start();
    #endif

    start_gpu(arr, spec_table, lg);
    
    #if PRINT_TIME
      float gpu_time = gpu_time_elapsed(gpu_timer);
      printf("\nGPU Time Elapsed: %1.8f seconds\n", gpu_time);
    #endif

  #endif
  

  #if ENABLE_SERIAL

    #if PRINT_TIME
      clock_t cpu_timer = cpu_timer_start();
    #endif

    i_t outRow = mcmc_serial(arr, lg) - lg + endstates + 1;
  
    #if PRINT_RESULT
      printf("\nLG = %I64d FINISHED AT OUTROW # %I64d", lg, outRow);
    #endif

    #if PRINT_TIME
      float cpu_time = cpu_time_elapsed(cpu_timer);
      printf("\nCPU Time Elapsed: %1.3f seconds\n", cpu_time);
    #endif

  #endif

  return 0;
}


// printf("%lu", sizeof(i_t)); // Tells num bytes in data type
// printf("%d", PTRDIFF_MAX); // Used to determine practical max bytes per array

