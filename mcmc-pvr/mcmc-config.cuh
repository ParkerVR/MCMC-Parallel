// Markov Chain Monte Carlo Compilation Options
// This is the only file you should need to change!


// Choose compiler from list below
#define COMPILER 0

#define GCC        0
#define GPP        1
#define CUDA       2
#define OTHER      3


// Choose between a specific or random seed
#define SEEDED               0
#define SEED                 420


// Array value type
#define num_t                float

// Iterator type
#define i_t                  int64_t


// Array Width/Height
// Max LG 23160 (2^14.5)
#define ARR_LG               (i_t)23160
#define ARR_SZ               (i_t)(ARR_LG*ARR_LG)

#define ENDSTATES            2





// Priting Options
#define PRINT_ARR            0
#define PRINT_N_DECIMALS     3

#define PRINT_TIME           1
#define PRINT_STEPS          0
#define PRINT_RESULT         1

#define ENABLE_SERIAL        1
#define ENABLE_GPU           0