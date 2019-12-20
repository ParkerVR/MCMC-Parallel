// Markov Chain Monte Carlo Compilation Options
// This is the only file you should need to change!


// Select your compiler id from list below
#define COMPILER   2

#define GCC        0
#define GPP        1
#define NVCC       2
#define OTHER      3


// Choose between a specific or random seed
#define SEEDED               1
#define SEED                 420


// Array value type
#define num_t                float

// Iterator type
#define i_t                  int64_t


// Array Width/Height
// Max LG 23160   (2^14.5)  cyg
// Max LG 111111  (2^16.75) zsh64
// Max LG >1000000          scc
// For GPU, ARR_LG must be at least 8
#define ARR_LG               (i_t)8
#define ARR_SZ               (i_t)(ARR_LG*ARR_LG)

#define ENDSTATES            2



// Threads per block should be a multiple of 32; TPB = TPBL^2
#define THREADS_PER_BLOCK_L  8
// Number of blocks 
#define NUM_BLOCKS_L         1


// Priting Options
#define PRINT_ARR            1
#define PRINT_N_DECIMALS     3

#define PRINT_TIME           1
#define PRINT_STEPS          1
#define PRINT_RESULT         1

#define ENABLE_SERIAL        1
#define ENABLE_GPU           1