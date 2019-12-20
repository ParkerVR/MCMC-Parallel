// MCMC Parallel Algorithm (CUDA)


void prep_gpu(num_t* form_arr, I_t gpu_lg) {

  I_t gpu_sz = gpu_lg*gpu_lg;

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