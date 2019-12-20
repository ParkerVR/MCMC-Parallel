/*
   CUDA MCMC program, intended just to test ability
   to compile and run a basic CUDA MCMC and compare to serial

     nvcc mcmc_v1.cu -o mcmc_v1
*/

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
//#include "cuPrintf.cu"
//#include "cuPrintf.cuh"

void initializeArray1D(float *arr, int sz, int seed);
void printarr(float *arr, int sz);
/*
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
*/
// L signifies per length (X or Y) as everything is a square!

#define ARR_L                16
#define ARR                  ARR_L*ARR_L
#define THREADS_PER_BLOCK_L  16
#define THREADS_PER_BLOCK    THREADS_PER_BLOCK_L*THREADS_PER_BLOCK_L
#define NUM_BLOCKS_L         ARR_L/THREADS_PER_BLOCK_L
#define SIZE                 14


#define PRINT_TIME           1
#define ENABLE_SERIAL        1

#define IMUL(a, b) __mul24(a, b)


__global__ void kernel_mcmc (float* a, int sz, int* out) {


  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = by * blockDim.y + ty;
  const int j = bx * blockDim.x + tx;

  __shared__ float a_shared[THREADS_PER_BLOCK];
  __shared__ float b_shared[THREADS_PER_BLOCK];

  __shared__ float curand[32];

  float sum0 = 0;
  float sum1 = 0;
  float sum2 = 0;
  float sum3 = 0;
  int index, m;
  int curand_index
  
  if ( threadJob == 0 ) { // Denotes the job of random generation
    curandGen( curand, curand_index );
  } else if ( threadJob == 1 ) { // Denotes the job of tracing DP
    trace_index = findTraceIndex(trace, row);
    getNext( trace, trace_index, curand[curand_index] );
  } else if ( threadJob == 2 ) { // Denotes the 'real' trace job
    nextRow = followTrace( trace, row );
  }


  for (m = 0; m < NUM_BLOCKS_L; m++) {
    a_shared[ty*THREADS_PER_BLOCK_L+tx] = a[i*ARR_L + (m*THREADS_PER_BLOCK_L + tx)];
    b_shared[ty*THREADS_PER_BLOCK_L+tx] = b[j + (m*THREADS_PER_BLOCK_L + ty)*ARR_L];
    __syncthreads();

    for (index = 0; index < THREADS_PER_BLOCK_L; index+=4){
      sum0 += a_shared[ty*THREADS_PER_BLOCK_L+(index+0)] * b_shared[(index+0)*THREADS_PER_BLOCK_L+tx];
      sum1 += a_shared[ty*THREADS_PER_BLOCK_L+(index+1)] * b_shared[(index+1)*THREADS_PER_BLOCK_L+tx];
      sum2 += a_shared[ty*THREADS_PER_BLOCK_L+(index+2)] * b_shared[(index+2)*THREADS_PER_BLOCK_L+tx];
      sum3 += a_shared[ty*THREADS_PER_BLOCK_L+(index+3)] * b_shared[(index+3)*THREADS_PER_BLOCK_L+tx];
    }
    __syncthreads();
  }

  p[i*ARR_L+j] = sum0+sum1+sum2+sum3;
  
} 
 
void serial_mcmc (float* arr, int sz, int* out) {
  
  int i, j;
  int row = 0;
  float random;

  int index = 0;
  while( arr[row*sz+row] != 1.0 ) {
    random = (float)rand()/(float)(RAND_MAX);
    for(j = 0; j < sz; j++) {
      if( arr[row*sz+j] >= random ) {
        row = j;
        break;
      }
    }
  }

  out = &index;
}

void convert_threadable(float* a, int sz, float* p) {

  size_t allocSize = ARR * sizeof(float);
  p = (float *) malloc(allocSize);
  
  int i;
  int j;
  for (i = 0; i < sz; i++) {
    for (j = 0; j < sz; j++) {
      p[i*ARR_L+j] = a[i*sz+j];
    }
  }

}

void get_tolerance (float* a, float* b, float* max_tol_pct, float* max_tol) {
  
  int i, j;
  float temp_tol, temp_tol_pct, avg;

  for(i = 0; i < ARR_L; i++) {
    for(j = 0; j < ARR_L; j++) {
      //printf("%f,%f\n",a[i*ARR_L+j], b[i*ARR_L+j]);
      
      temp_tol = a[i*ARR_L+j] - b[i*ARR_L+j];
      if (temp_tol < 0.0) {
        temp_tol = 0.0-temp_tol;
      }
      avg = (a[i*ARR_L+j] + b[i*ARR_L+j])/2;
      temp_tol_pct = temp_tol/avg;
      if (temp_tol_pct > *max_tol_pct) {
        *max_tol_pct = temp_tol_pct; 
        *max_tol = temp_tol;
      }
    }
  }      
}


int main(int argc, char **argv) {

  /*
  // GPU Timing variables
  cudaEvent_t start_g_data,   stop_g_data;
  cudaEvent_t start_g_kernel, stop_g_kernel;
  float       elapsed_g_data, elapsed_g_kernel;
*/
  // CPU Timing variables
  time_t start_c, stop_c;
  float elapsed_c;

  // Arrays on GPU global memory
  float *d_a;

  // Arrays on the host memory
  float *h_a;

/*


  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

  // Allocate arrays on GPU memory
  size_t allocSize = ARR * sizeof(float);

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_a, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_p, allocSize));


  // Allocate arrays on host memory
  h_a = (float *) malloc(allocSize);
  h_b = (float *) malloc(allocSize);
  h_p = (float *) malloc(allocSize);


  // Arrays are initialized with known seeds for reproducability
  initializeArray1D(h_a, ARR, 420);
  initializeArray1D(h_b, ARR, 69);
  //int i;
  //for(i = 0; i < 50; i++)
    //printf("(%f, %f)\n", h_a[i], h_b[i]);

  printf("\n Beginning tests with %d X %d matrices\n", ARR_L, ARR_L);


#if PRINT_TIME
  // Create the gpu data transfer event timer
  cudaEventCreate(&start_g_data);
  cudaEventCreate(&stop_g_data);
  // Record event on the default stream
  cudaEventRecord(start_g_data, 0);
#endif

  // Transfer the arrays to the GPU memory
  CUDA_SAFE_CALL(cudaMemcpy(d_a, h_a, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_b, h_b, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_p, h_p, allocSize, cudaMemcpyHostToDevice));

  dim3 dimGrid(NUM_BLOCKS_L, NUM_BLOCKS_L, 1);
  dim3 dimBlock(THREADS_PER_BLOCK_L, THREADS_PER_BLOCK_L, 1);
  
#if PRINT_TIME
  // Create the gpu kernel event timer
  cudaEventCreate(&start_g_kernel);
  cudaEventCreate(&stop_g_kernel);
  // Record event on the default stream
  cudaEventRecord(start_g_kernel, 0);
#endif  
  

  // Runs Kernels
  kernel_mmm<<<dimGrid, dimBlock>>>(d_a, d_b, d_p);
  
  // Check for errors during launch
  CUDA_SAFE_CALL(cudaPeekAtLastError());



#if PRINT_TIME
  // Stop and destroy the gpu kernel timer
  cudaEventRecord(stop_g_kernel,0);
  cudaEventSynchronize(stop_g_kernel);
  cudaEventElapsedTime(&elapsed_g_kernel, start_g_kernel, stop_g_kernel);
  printf("\nGPU kernel time: %f (msec)\n", elapsed_g_kernel);
  cudaEventDestroy(start_g_kernel);
  cudaEventDestroy(stop_g_kernel);
#endif

  // Transfer the results back to the host
  CUDA_SAFE_CALL(cudaMemcpy(h_p, d_p, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
  // Stop and destroy the gpu data transfer timer
  cudaEventRecord(stop_g_data,0);
  cudaEventSynchronize(stop_g_data);
  cudaEventElapsedTime(&elapsed_g_data, start_g_data, stop_g_data);
  printf("\nGPU data transfer time: %f (msec)\n", elapsed_g_data);
  cudaEventDestroy(start_g_data);
  cudaEventDestroy(stop_g_data);
#endif

  // Free-up device memory
  CUDA_SAFE_CALL(cudaFree(d_a));
  CUDA_SAFE_CALL(cudaFree(d_b));
  CUDA_SAFE_CALL(cudaFree(d_p));
*/

  // Runs serial code
#if ENABLE_SERIAL
  // Array for serial
  float *s_p;
  int sz = SIZE;

  size_t allocSizeSerial = sz*sz* sizeof(float);

  // Allocate array on host memory
  s_p = (float *) malloc(allocSizeSerial);


  initializeArray1D(s_p, sz, 1);

  // Conduct serial SOR
  int* out;
  
  start_c = time(NULL);
  serial_mcmc(s_p, sz, out);
  stop_c = time(NULL);
  elapsed_c = difftime(stop_c,start_c);
  int outVal = *out;
  printf("Output row: %d\n", outVal);
  printf("\nCPU time: %f (sec)\n", elapsed_c);

  /*
  float max_tol = 0;
  float max_tol_pct = 0;
  get_tolerance(h_p, s_p, &max_tol_pct, &max_tol);
  printf("\nMAXIMUM TOLERANCE: %f%, %f\n", max_tol_pct, max_tol);
  */

  free(s_p);
#endif
/*
  // Free-up host memory
  free(h_a);
  free(h_b);
  free(h_p);
*/
  return 0;

}


void initializeArray1D(float *arr, int sz, int seed) {
  int i, j;
  float randNum;
  srand(seed);

  // Generate random rows that sum to probability of 1

  float rowSum;
  for (i = 0; i < sz-2; i++) {
    rowSum = 0;
    for (j = 0; j < sz; j++) {
      randNum = (float) rand();
      rowSum += randNum;
      arr[i*sz + j] = rowSum;
      printf("%f\n", arr[i*sz+j]);
    }
    for (j = 0; j < sz; j++) {
      arr[i*sz + j] = arr[i*sz + j] / rowSum;
    }
  }

  // Define (2) end states
  // Note based on rules currently, last number will always be end state
  for (int j = 0; j < sz; j++) {
    arr[(sz-2)*sz + j] = 0;
    arr[(sz-1)*sz + j] = 0;
  }

  arr[(sz-2)*sz + sz-2] = 1.0;
  arr[(sz-1)*sz + sz-1] = 1.0;

  printarr(arr,sz);
}

void printarr(float* arr, int sz) {
  int i, j;
  for(i = 0; i < sz; i++){
    for(j = 0; j < sz; j++) {
      printf("%f, ", arr[i*sz+j] );
    }
    printf("\n");
  }
}