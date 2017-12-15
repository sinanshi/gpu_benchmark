/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

//#define DOUBLE__PRECISION
#ifdef DOUBLE__PRECISION
    #define real double
    #define GEMM cublasDgemm
#else
    #define real float
    #define GEMM cublasSgemm
#endif

#undef VERIFY 
/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, real alpha, const real *A, const real *B,
    real beta, real *C)
{
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      real prod = 0;

      for (k = 0; k < n; ++k)
      {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}


int gpu_gemm(const real *h_A, const real *h_B, real *h_C, const real alpha,
    const real beta, const int N)
{
  real *d_A = 0;
  real *d_B = 0;
  real *d_C = 0;
  int n2 = N * N;

  cublasStatus_t status;
  cublasHandle_t handle;

  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }
  /* Allocate device memory for the matrices */
  if (cudaMalloc((void **)&d_A, n2 * sizeof(d_A[0])) != cudaSuccess)
  {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return EXIT_FAILURE;
  }
  if (cudaMalloc((void **)&d_B, n2 * sizeof(d_B[0])) != cudaSuccess)
  {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc((void **)&d_C, n2 * sizeof(d_C[0])) != cudaSuccess)
  {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);
  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);
  status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

  /* Performs operation using cublas */
  status = GEMM(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }

  /* Read the result back */
  status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

  if (cudaFree(d_A) != cudaSuccess)
  {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_B) != cudaSuccess)
  {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_C) != cudaSuccess)
  {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  return 0;
}


static int benchmark_blas(const int N)
{
  real *h_A;
  real *h_B;
  real *h_C;
  real *h_C_ref;
  real alpha = 1.0f;
  real beta = 0.0f;
  int n2 = N * N;
  int i;
  real error_norm;
  real ref_norm;
  real diff;



  /* Allocate host memory for the matrices */
  h_A = (real *)malloc(n2 * sizeof(h_A[0]));
  h_B = (real *)malloc(n2 * sizeof(h_B[0]));
  h_C = (real *)malloc(n2 * sizeof(h_C[0]));
  h_C_ref = (real *)malloc(n2 * sizeof(h_C[0]));

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++)
  {
    h_A[i] = rand() / (real)RAND_MAX;
    h_B[i] = rand() / (real)RAND_MAX;
    h_C[i] = rand() / (real)RAND_MAX;
    h_C_ref[i] = h_C[i];
  }

#ifdef VERIFY
  /* Performs operation using plain C code*/ 
  clock_t c_start = clock();
  simple_sgemm(N, alpha, h_A, h_B, beta, h_C_ref);
  clock_t c_end = clock();
#endif

  clock_t g_start = clock();
  gpu_gemm(h_A, h_B, h_C, alpha, beta, N);
  clock_t g_end = clock();
  double g_time = (double)(g_end - g_start) / CLOCKS_PER_SEC;
  std::cout << N << " " << g_time << " "<< 2.0 * pow(N, 3) / g_time / 1000 /1000 / 1000;

#ifdef VERIFY
    std::cout<<" "<< 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << std::endl;
#else
    std::cout << std::endl;
#endif


#ifdef VERIFY
  error_norm = 0;
  ref_norm = 0;

  for (i = 0; i < n2; ++i)
  {
    diff = h_C_ref[i] - h_C[i];
    error_norm += diff * diff;
    ref_norm += h_C_ref[i] * h_C_ref[i];
  }

  error_norm = (real)sqrt((double)error_norm);
  ref_norm = (real)sqrt((double)ref_norm);

  if (fabs(ref_norm) < 1e-7)
  {
    fprintf(stderr, "!!!! reference norm is 0\n");
    return EXIT_FAILURE;
  }
  if (error_norm / ref_norm > 1e-6f)
  {
    printf("simpleCUBLAS test failed.\n");
    exit(EXIT_FAILURE);
  }
#endif


  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);


}


/* Main */
int main(int argc, char **argv)
{
  int i, N;
  int dev = 0; 

  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  int MAX_N = (int)(sqrt(deviceProp.totalGlobalMem/3.0/sizeof(real)));
  std::cout<<deviceProp.name<<"    "<<MAX_N<<std::endl;
  if (dev == -1)
  {
    return EXIT_FAILURE;
  }

  /* Initialize CUBLAS */
  printf("Matmul test: \n");
  N = 1000;
  for(i = 0; i < 20; ++i)
  {
    benchmark_blas(N);
    N = N + 1000;
  }

}
