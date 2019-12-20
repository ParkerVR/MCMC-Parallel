// MCMC Parallel Algorithm (CUDA)


// Sets up gpu to be run
void prep_gpu(num_t* form_arr, i_t gpu_lg);

// Kernel
__global__ void kernel_mcmc (num_t* arr, i_t gpu_lg, i_t lg, int* out);


void prep_gpu(num_t* form_arr, i_t gpu_lg) {

  i_t gpu_sz = gpu_lg*gpu_lg;

  // Array on GPU global memory
  num_t* gpu_arr;

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate array on GPU memory
  size_t allocSize = gpu_sz * sizeof(num_t);
  CUDA_SAFE_CALL(cudaMalloc((void **)&gpu_arr, allocSize));

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(gpu_arr, form_arr, allocSize, cudaMemcpyHostToDevice));

  dim3 dimGrid(NUM_BLOCKS_L, NUM_BLOCKS_L, 1);
  dim3 dimBlock(THREADS_PER_BLOCK_L, THREADS_PER_BLOCK_L, 1);

}


__global__ void kernel_mcmc (num_t* arr, I_t gpu_lg, I_t lg, int* out) {

  const i_t bx = blockIdx.x;
  const i_t by = blockIdx.y;
  const i_t tx = threadIdx.x;
  const i_t ty = threadIdx.y;

  const i_t i = by * blockDim.y + ty;
  const i_t j = bx * blockDim.x + tx;

}

