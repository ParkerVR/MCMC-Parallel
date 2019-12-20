// MCMC Timer Tools


#if PRINT_TIME == 1

  // CPU Timers
  clock_t cpu_timer_start();
  float cpu_time_elapsed(clock_t start_c);

  clock_t cpu_timer_start() {

    clock_t start_c = clock();
    return start_c;

  }

  // Returns elapsed time in seconds
  float cpu_time_elapsed(clock_t start_c) {

    clock_t stop_c = clock();
    float elapsed_c = (float)(stop_c - start_c) / CLOCKS_PER_SEC;  // seconds 
    return elapsed_c;

  }


// GPU Timers
#if ENABLE_GPU == 1

  struct cudaTimer { 
    cudaEvent_t start_g, stop_g; 
  }; 
  typedef struct cudaTimer time_g; 
  
  time_g gpu_timer_start();
  float gpu_time_elapsed(time_g gpu_timer);

  time_g gpu_timer_start() {
    
    //cudaEvent_t start_g, stop_g;
    time_g gpu_timer;
    gpu_timer.start_g = start_g;
    gpu_timer.stop_g = stop_g;
    cudaEventCreate(&gpu_timer.start_g);
    cudaEventCreate(&gpu_timer.stop_g);
    cudaEventRecord(gpu_timer.start_g, 0);

    return gpu_timer;

  }

  // Returns elapsed time in seconds
  float gpu_time_elapsed(time_g gpu_timer) {

    float elapsed_g;
    cudaEventRecord(gpu_timer.stop_g, 0);
    cudaEvenSynchronize(gpu_timer.stop_g);
    cudaEventElapsedTime(&elapsed_g, gpu_timer.start_g, gpu_timer.stop_g); // ms

    elapsed_g = elapsed_g/1000; // seconds

    cudaEventDestroy(gpu_timer.start_g);
    cudaEventDestroy(gpu_timer.stop_g);

    return elapsed_g;
    
  }

#endif
#endif