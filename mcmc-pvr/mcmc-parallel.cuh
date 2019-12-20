// MCMC Parallel Algorithm (CUDA)
 
// cuRand device api: https://docs.nvidia.com/cuda/curand/device-api-overview.html

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


// Sets up gpu to be run
void start_gpu(num_t* form_arr, num_t* spec_table, i_t gpu_lg);

// Kernel
__global__ void kernel_mcmc (num_t* arr, i_t gpu_lg, i_t lg, num_t* spec_arr, i_t seed, num_t* rand_arr, i_t rand_lg);
__device__ i_t checkFinishedTr(num_t* arr, i_t lg, i_t row);
__device__ i_t nextRowTr(num_t* arr, i_t lg, i_t row, num_t rand);
__device__ curandState prep_rand(i_t seed, i_t thread_id, i_t offset);
__device__ void gen_rand(num_t rand_arr, i_t rand_lg, curandState s);
__device__ i_t spec_search(num_t* spec_arr, num_t rand);

void start_gpu(num_t* form_arr, num_t* spec_table, i_t gpu_lg) {

  i_t gpu_sz = gpu_lg*gpu_lg;
  i_t spec_lg = ARR_LG - ENDSTATES;

  // Arrays on GPU global memory
  num_t* gpu_arr;
  num_t* rand_arr;
  num_t* spec_arr;
  
  i_t rand_lg = 100;

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate array on GPU memory
  size_t allocSize = gpu_sz * sizeof(num_t);
  size_t allocSizeRand = rand_lg * sizeof(num_t);
  size_t allocSizeSpec = spec_lg * sizeof(num_t);
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_arr, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&rand_arr, allocSizeRand));
  CUDA_SAFE_CALL(cudaMalloc((void **)&spec_arr, allocSizeSpec));

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(gpu_arr, form_arr, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(spec_arr, spec_table, allocSizeSpec, cudaMemcpyHostToDevice));
  
  

  dim3 dimGrid(NUM_BLOCKS, 1, 1);
  dim3 dimBlock(THREADS_PER_BLOCK, 1, 1);

  #if PRINT_TIME
    time_g gpu_timer_kernel = gpu_timer_start();
  #endif

  cudaPrintfInit();
  kernel_mcmc<<<dimGrid, dimBlock>>>(gpu_arr, gpu_lg, gpu_lg, spec_arr, SEED, rand_arr, rand_lg);
  
  cudaDeviceSynchronize();

  #if PRINT_TIME
    float gpu_time_kernel = gpu_time_elapsed(gpu_timer_kernel);
    printf("\nGPU Kernel Time: %1.8f seconds\n", gpu_time_kernel);
  #endif

  
  cudaPrintfDisplay(stdout,true);
  cudaPrintfEnd();

}

__device__ curandState prep_rand(i_t seed, i_t thread_id, i_t offset) {

  curandState s;

  // seed a random number generator
  curand_init(seed+thread_id, 0, offset, &s);

  return s;

}

__device__ void gen_rand(num_t* rand_arr, i_t rand_lg, curandState s) {
  i_t i;
  for(i = 0; i < rand_lg; i++)
    rand_arr[i] = curand_uniform(&s);
}


__global__ void kernel_mcmc (num_t* arr, i_t gpu_lg, i_t lg, num_t* spec_table, i_t seed, num_t* rand_arr, i_t rand_lg) {

  i_t id = threadIdx.x * blockDim.x;
  /* Each thread gets same seed, a different sequence 
      number, no offset */

  __shared__ i_t index_main;

  
  i_t index = 0;

  i_t rand_index = 0;

  i_t offset = 0;

  i_t step_count = 0;
  curandState s = prep_rand(seed, id, offset);
  gen_rand(rand_arr, rand_lg, s);

  i_t start_pos;

  __shared__ i_t cur_pos [THREADS_PER_BLOCK-1];
  __shared__ i_t step_counts [THREADS_PER_BLOCK-1];
  __shared__ i_t alert;
  

  if (id == 0) {
    index_main = 0;
    alert = 0;
    while( !checkFinishedTr(arr, lg, index_main) ) {
      if (alert > 0){
        index_main = cur_pos[index-1];
        step_count += step_counts[index-1];
        alert = 0;
      }

      index_main = nextRowTr(arr, lg, index_main, rand_arr[rand_index]);
      rand_index++;
      step_count++;

      // random refresh
      if (rand_index == rand_lg){
        offset += rand_lg;
        s = prep_rand(seed, id, offset);
        gen_rand(rand_arr, rand_lg, s);
        rand_index = 0;
      }
      __syncthreads();
    }
    
    cuPrintf("\nParallel complete in %I64d steps.", step_count);
  } else if (id > 0) {    
    index = spec_search(spec_table, rand_arr[rand_index]);
    start_pos = index;
    while( !checkFinishedTr(arr, lg, index_main) ) {
      index = nextRowTr(arr, lg, index, rand_arr[rand_index]);
      rand_index++;
      step_count++;
      
      // random refresh
      if (rand_index == rand_lg){
        offset += rand_lg;
        s = prep_rand(seed, id, offset);
        gen_rand(rand_arr, rand_lg, s);
        rand_index = 0;
      }

      if(start_pos == index_main && alert == 0 || index > cur_pos[alert-1]) {
        alert = id;
        cur_pos[id-1] = index;
        step_counts[id-1] = step_count;
      }
      
      __syncthreads();
    }
  } else {
    while( !checkFinishedTr(arr, lg, index_main) ) {

      // random refresh
      if (rand_index == rand_lg) {
        offset += rand_lg;
        s = prep_rand(seed, id, offset);
        gen_rand(rand_arr, rand_lg, s);
        rand_index = 0;
      }

      __syncthreads();
    } 
  }

}

__device__ i_t checkFinishedTr(num_t* arr, i_t lg, i_t row) {
  if (arr[row*lg+row] == 1.0)
    return 1;
  else return 0;
}


__device__ i_t nextRowTr(num_t* arr, i_t lg, i_t row, num_t random) {

  i_t index;
 
  index = lg*random;

  while (index != lg-1 && random > arr[lg*row + index])
    index++;

  while (index != 0 && random < arr[lg*row + index-1])
    index--;

#if PRINT_STEPS  
  cuPrintf("\n Current Row: %I64d", row);
  cuPrintf("\n Random: %1.3f", random);
  cuPrintf("\n Next Row: %I64d\n", index);
#endif

  return index;
}


__device__ i_t spec_search(num_t* spec_arr, num_t random) {
  i_t spec_lg = ARR_LG-ENDSTATES;
  i_t index = spec_lg*random;

  while (index != spec_lg-1 && random > spec_arr[index])
    index++;

  while (index != 0 && random < spec_arr[index-1])
    index--;

  return index;
}
