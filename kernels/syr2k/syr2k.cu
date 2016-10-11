/**
 * syr2k.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include <polybenchUtilFuncts.h>
#include <ca.h>


//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#define N 2048
#define M 2048

/* Problem size */

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 12435
#define BETA 4546

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

#define ITEMS N * M 
#define FIELDS 2

typedef struct data_item_type {
  DATA_TYPE a;
  DATA_TYPE c;
} data_item;

#define SPARSITY N
#define TILEFACTOR 1
#define TILE 4 // ITEMS_PER_REF_GROUP / TILEFACTOR    // Tile is a factor of ITEMS_PER_REF_GROUP

#if 0
void convert_aos_to_ca(data_item *aos_data, DATA_TYPE *ca) {
  int set_count = 0;
  for (int j = 0, t = 0; j < ITEMS * FIELDS; j += TILE * FIELDS, t++)
    for (int i = 0, m = j; i < TILE; i++, m++) {
      if (t == REF_GROUP_RESET) {
	set_count++;
	t = 0;
      }
      int aos_index = ((t * TILE + i) * SPARSITY) + set_count; 
      for (int p = 0; p < FIELDS; p++) {
	ca[m + TILE * p] = *((DATA_TYPE *) (aos_data + aos_index) + p);
      }
#ifdef DEBUG       
      printf("%d\t%d\t%d\n", j, m, aos_index);
#endif
    }
}
#endif



void init_arrays(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j;
  
	for (i = 0; i < N; i++)
    	{
    		for (j = 0; j < N; j++)
		{
			C[i*N + j] = ((DATA_TYPE) i*j + 2) / N;
		}
      	
		for (j = 0; j < M; j++)
		{
	  		A[i*N + j] = ((DATA_TYPE) i*j) / N;
	  		B[i*N + j] = ((DATA_TYPE) i*j + 1) / N;
		}
    	}
}


void syr2k(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j, k;
		
  	for (i = 0; i < N; i++)
	{
   		for (j = 0; j < N; j++)
		{
     			C[i*N + j] *= BETA;
		}
	}

  	for (i = 0; i < N; i++)
	{
   		for (j = 0; j < N; j++)
		{
      			for (k = 0; k < M; k++)
			{
	  			C[i*N + j] += ALPHA * A[i*M + k] * B[j*M + k];
	 		 	C[i*N + j] += ALPHA * B[i*M + k] * A[j*M + k];
			}
		}
	}
}


void compareResults(DATA_TYPE *C, DATA_TYPE *C_outputFromGpu)
{
	int i,j,fail;
	fail = 0;

	// Compare C with D
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			if (percentDiff(C[i*N + j], C_outputFromGpu[i*N + j]) > PERCENT_DIFF_ERROR_THRESHOLD)
			{ 
				fail++;
				//				printf("%3.2f\t%3.2f\n",C[i*N+j],C_outputFromGpu[i*N+j]);
			}
		}
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	//	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void syr2k_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < N) && (j < N))
	{
	  c[i * N + j] *= BETA;
	  
	  int k;
	  for(k = 0; k < M; k++)
	    {
	      c[i * N + j] += ALPHA * a[i * M + k] * b[j * M + k] + ALPHA * b[i * M + k] * a[j * M + k];
	    }
	}
}


__global__ void syr2k_kernel_ca(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *ca)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	int j = 0;
	for (j = 0; j < N; j++ )
 	if ((i < N) && (j < N))
	{
	  
	  int index = (i * FIELDS) - (i % TILE) * (FIELDS - 1);
	  int j_new  = (j * FIELDS) - (j % TILE) * (FIELDS - 1);
	  int j_index = (j * SPARSITY * FIELDS); 
	  ca[index + j_index] *= BETA;
	  int k;
	  for(k = 0; k < M; k++) {
	    int k_index = (k * SPARSITY * FIELDS); 
	    ca[index + j_index] += ALPHA * a[i * M + k] * b[j * M + k] 
	      + ALPHA * b[i * M + k] * a[j * M + k];
	  }
	}
}


void syr2kCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* C_outputFromGpu, DATA_TYPE *ca) 
{
	double t_start, t_end;

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *ca_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * M);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * M);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * N * N);

	cudaMalloc((void **)&ca_gpu, sizeof(DATA_TYPE) * N * N);

	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);

	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(ca_gpu, ca, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_X) ), (size_t)(ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_Y) )));
	
	t_start = rtclock();
#ifdef DEFAULT 
	syr2k_kernel<<<grid,block>>>(A_gpu,B_gpu,C_gpu);
#endif
#ifdef CA
	syr2k_kernel_ca<<<grid,block>>>(A_gpu,B_gpu,ca_gpu);
#endif
	cudaThreadSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	
#ifdef DEFAULT 
	cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
#endif
#ifdef CA
	cudaMemcpy(C_outputFromGpu, ca_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);
#endif

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);
}


int main()
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* C_outputFromGpu;

	A = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

	C_outputFromGpu = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

	DATA_TYPE *ca;
	data_item *aos;

	ca = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE) * FIELDS);
	aos = (data_item *) malloc(N*M*sizeof(data_item) * FIELDS);

	init_arrays(A, B, C);
    
	for (int i = 0; i < N * M; i++) {
	  aos[i].a = A[i];
	  aos[i].c = C[i];
	}

	convert_aos_to_ca(aos, ca, ITEMS, FIELDS, TILE, SPARSITY);
	check_ca_conversion(A, ca, ITEMS, FIELDS, TILE, SPARSITY);

	GPU_argv_init();
	syr2kCuda(A, B, C, C_outputFromGpu, ca);
	
	t_start = rtclock();
	syr2k(A, B, C);
	t_end = rtclock();
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(C, C_outputFromGpu);

	free(A);
	free(B);
	free(C);
	free(C_outputFromGpu);

  	return 0;
}

