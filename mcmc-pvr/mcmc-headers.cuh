// Lists all internal headers

// Hierarchy (Contains all Sys Includes)
#include "./headers/mcmc-compilers.cuh" 

// Tools
#include "./headers/mcmc-arr-tools.cuh"
#include "./headers/mcmc-timer-tools.cuh"

// Algorithms
#if ENABLE_SERIAL
  #include "mcmc-serial.cuh"
#endif

#if ENABLE_GPU
  #include "mcmc-parallel.cuh"
#endif