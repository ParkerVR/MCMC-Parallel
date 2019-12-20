// Markov Chain Monte Carlo Compilation Options- the only file that will need to be changed!


// Choose compiler from list below
#define COMPILER 0

#define GCC        0
#define GPP        1
#define CUDA       2
#define OTHER      3

// Array Width/Height
#define ARR_LG               4
#define ARR_SZ               ARR_LG*ARR_LG

#define ENDSTATES            2

#define num_t                float

#define PRINT_TIME           1
#define PRINT_N_DECIMALS     3
#define ENABLE_SERIAL        1
#define ENABLE_GPU           0