// Compiler Based Include List

// gcc -x c mcmc-top.cu -o mcmc_s
#if COMPILER == GCC
  #include <stdlib.h>
  #include <stdio.h>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>


// g++ -x c mcmc-top.cu -o mcmc_s
#elif COMPILER == GPP
  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>
  
// nvcc mcmc-top -o mcmc_c
#elif COMPILER == NVCC
  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>
  #include <curand.h>
  #include <curand_kernel.h>

#else
  #include <cstdio>
  #include <cstdlib>
  #include <math.h>
  #include <time.h>
  #include <stdint.h>

#endif


#if ENABLE_GPU
  #include "cuPrintf.cu"
  #include "cuPrintf.cuh"
#endif