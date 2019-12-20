#include <time.h>



int main() {


float* d_a;
// my array initialized
initializeArray1D(d_a, ARR_LG, 420);


#if ENABLE_GPU


  // GPU Timing variables
    cudaEvent_t start_g_data,   stop_g_data;
    cudaEvent_t start_g_kernel, stop_g_kernel;
    float       elapsed_g_data, elapsed_g_kernel;


  // <set stuff up not timed>

  // Create the gpu data transfer event timer
    cudaEventCreate(&start_g_data);
    cudaEventCreate(&stop_g_data);
  // Record event on the default stream
   cudaEventRecord(start_g_data, 0);

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, allocSize, cudaMemcpyHostToDevice));

  
  // Create the gpu kernel event timer
    cudaEventCreate(&start_g_kernel);
    cudaEventCreate(&stop_g_kernel);
    // Record event on the default stream
    cudaEventRecord(start_g_kernel, 0);

  // Runs Kernels
  kernel_mmm<<<dimGrid, dimBlock>>>(d_a, d_b, d_p);
  // < do serial stuff here >

  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());  

  // Stop and destroy the gpu kernel timer
    cudaEventRecord(stop_g_kernel,0);
    cudaEventSynchronize(stop_g_kernel);
    cudaEventElapsedTime(&elapsed_g_kernel, start_g_kernel, stop_g_kernel);
    cudaEventDestroy(start_g_kernel);
    cudaEventDestroy(stop_g_kernel);

  // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_p, d_p, allocSize, cudaMemcpyDeviceToHost));

  // Stop and destroy the gpu data transfer timer
    cudaEventRecord(stop_g_data,0);
    cudaEventSynchronize(stop_g_data);
    cudaEventElapsedTime(&elapsed_g_data, start_g_data, stop_g_data);
    cudaEventDestroy(start_g_data);
    cudaEventDestroy(stop_g_data);

  // < do output stuff not timed >
  CUDA_SAFE_CALL(cudaFree(d_a));
  printf("Output row: %d\n", out);

  // Print time
  printf("\nGPU kernel time: %f (msec)\n", elapsed_g_kernel);
  printf("\nGPU data transfer time: %f (msec)\n", elapsed_g_data);

#if ENABLE_SERIAL

  // CPU Timing variables
    time_t start_c, stop_c;
    float elapsed_c;

  
  // < set up stuff not timed >

  // Get time start
    start_c = time(NULL);
    
  serial_mcmc(d_a, sz, out);
  // < do serial stuff here >

  // Get time end
    stop_c = time(NULL);
    elapsed_c = difftime(stop_c,start_c);

  // < do output stuff not timed >
  free(s_p);
  free(start_c);
  free(start_c);
  printf("Output row: %d\n", out);

  // Print time
  printf("\nCPU time: %f (sec)\n", elapsed_c);
  

#endif


  return 0;
}