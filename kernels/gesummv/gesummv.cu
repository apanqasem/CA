/**
 * gesummv.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 * scalar, vector and matrix multiplication 
 * 
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include <polybenchUtilFuncts.h>

#include <util.h>
#include <ca.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 43532.0f
#define BETA 12313.0f


/* ******************************************************************
 *  
 * CA Specific Definitions
 * Need to develop a method to extract this information from source
 * 
 * ******************************************************************/


#define ITEMS N * N
#define FIELDS 2
typedef struct data_item_type {
  DATA_TYPE a;
  DATA_TYPE b;
} data_item;

#define SPARSITY N



void gesummv(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
  int i, j;
  
  for (i = 0; i < N; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (j = 0; j < N; j++) {
      tmp[i] = A[i*N + j] * x[j] + tmp[i];
      y[i] = B[i*N + j] * x[j] + y[i];
#ifdef DEBUG
      if (i < 4) 
	printf("%d\t", i * N + j);
#endif
    }
#ifdef DEBUG
    if (i < 4) 
      printf("\n");
#endif
    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}

__global__ void gesummv_ca_kernel(DATA_TYPE *ca, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < N) {
    int j;
    int index = (i * FIELDS) - (i % TILE) * (FIELDS - 1);
    
    for(j = 0; j < SPARSITY; j++) {	
      tmp[i] += ca[index + (j * SPARSITY * FIELDS)] * x[j];
      y[i] += ca[index + (j * SPARSITY * FIELDS) + TILE] * x[j];
    }
    y[i] = ALPHA * tmp[i] + BETA * y[i];
  }
}


void init(DATA_TYPE* A, DATA_TYPE* x)
{
  	int i, j;
    	for (i = 0; i < N; i++)
    {
    	x[i] = ((DATA_TYPE) i) / N;
      	
    		for (j = 0; j < N; j++) 
    		{
    			A[i*N + j] = ((DATA_TYPE) i*j) / N;
    		}
    }
}


void compareResults(DATA_TYPE* y, DATA_TYPE* y_outputFromGpu)
{
	int i, fail;
	fail = 0;
	
	for (i=0; i<(N); i++) 
	{
		if (percentDiff(y[i], y_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD) 
		{
		  printf("%3.2f\n",y_outputFromGpu[i]);
		  fail++;
		}
	}
	
	fprintf(stderr, "%s\n", (fail > 0 ? "FAILED (GPU)" : "PASSED (GPU)"));
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	//	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}

__global__ void gesummv_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j = 0; j < N; j++)
		{	
			tmp[i] += a[i * N + j] * x[j];
			y[i] += b[i * N + j] * x[j];
		}
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}


void gesummvCuda(DATA_TYPE *ca, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp, DATA_TYPE* y_outputFromGpu)
{
  //	double t_start, t_end;		

	DATA_TYPE *ca_gpu;
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaMalloc((void **)&ca_gpu, sizeof(DATA_TYPE) * N * N * FIELDS);
	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * N);
	cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * N);
	

	cudaMemcpy(ca_gpu, ca, sizeof(DATA_TYPE) * N * N * FIELDS, cudaMemcpyHostToDevice);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice);

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((unsigned int)ceil( ((float)N) / ((float)block.x) ), 1);

	double t = mysecond();
#ifdef DEFAULT
		gesummv_kernel<<< grid, block>>>(A_gpu,B_gpu,x_gpu, y_gpu, tmp_gpu);
#endif
#ifdef CA 
	gesummv_ca_kernel<<<grid, block>>>(ca_gpu,x_gpu, y_gpu, tmp_gpu);
#endif
	cudaThreadSynchronize();
	t = 1.0E6 * (mysecond() - t);
	fprintf(stdout, "%3.2f\n", t/1000);
	cudaMemcpy(y_outputFromGpu, y_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost);


}


int main(int argc, char *argv[])
{

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* x;  
	DATA_TYPE* y;
	DATA_TYPE* y_outputFromGpu;
	DATA_TYPE* tmp;
	

	DATA_TYPE *ca;
	data_item *aos;
	ca = (DATA_TYPE*)malloc(ITEMS*sizeof(DATA_TYPE) * FIELDS);
	aos = (data_item *) malloc(ITEMS*sizeof(data_item));

	A = (DATA_TYPE*)malloc(ITEMS*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(ITEMS*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE)); 
	y = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_outputFromGpu = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

	init(A, x);

	for (int i = 0; i < ITEMS; i++) {
	  aos[i].a = A[i];
	  aos[i].b = B[i];
	}

	convert_aos_to_ca(aos, ca, ITEMS, FIELDS, SPARSITY);
	check_ca_conversion(A, ca, ITEMS, FIELDS, SPARSITY);

//	convert_aos_to_ca(aos, ca);

#ifdef DEBUG
	check_ca_conversion(A, ca);
#endif
	GPU_argv_init();
	gesummvCuda(ca, A, B, x, y, tmp, y_outputFromGpu);
	
	gesummv(A, B, x, y, tmp);
	
	compareResults(y, y_outputFromGpu);
	free(A);
	free(B);  
	free(x);  
	free(y);
	free(y_outputFromGpu);
	free(tmp);

	return 0;
}

