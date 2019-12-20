// Markov Chain Monte Carlo Compilation Options- the only file that will need to be changed!

#define ARR_L                16
#define ARR                  ARR_L*ARR_L
#define THREADS_PER_BLOCK_L  16
#define THREADS_PER_BLOCK    THREADS_PER_BLOCK_L*THREADS_PER_BLOCK_L
#define NUM_BLOCKS_L         ARR_L/THREADS_PER_BLOCK_L
#define size                 14


#define PRINT_TIME           1
#define ENABLE_SERIAL        1
#define ENABLE_GPU           0