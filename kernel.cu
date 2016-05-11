
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef enum {
	CONVERGENCE, CUDAFAIL, NOCONVERGENCE
} poison_status;


typedef struct t_problem_poison1d  {
	float tol, alpha, beta, a, b;
	unsigned int max_iter, n;

} problem_poison1d;

typedef struct t_solution_poison1d  {
	poison_status status;
	float * u;
} solution_poison1d;


__device__ int flag;


__device__ float f(float x) {
	return -12*x*x;
}

__global__ void solve_poison1d_kernel(const float h, float * u, const float tol, float a, float b, int n) {
	unsigned int i;
	float newValue, xi;

	i = threadIdx.x + blockIdx.x * blockDim.x+ 1;
	xi = a + i*(b - a) / (n + 1);

	newValue = 0.5*(u[i - 1] + u[i + 1] + h*h*(f(xi)));

	// Convergence test
	if (abs(u[i] - newValue) > tol) flag = 0;
	u[i] = newValue;
}

solution_poison1d solve_poison1d(problem_poison1d p, int * iter) {

	int  blockSize, minGridSize, gridSize, i, k, cpuConvergenceTest;
	float *dev_u, h;
	cudaError_t cudaStatus;

	solution_poison1d sp;
	h = (-p.a + p.b) / (p.n + 1);


	printf("h=%f\n", h);
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		sp.status = CUDAFAIL;
		return sp;
	}

	sp.u = (float*)malloc(sizeof(float)*(p.n + 1));
	cudaStatus = cudaMalloc((void**)&dev_u, sizeof(float)*(p.n + 1));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		sp.status = CUDAFAIL;
		return sp;
	}
	cudaStatus = cudaMalloc((void**)&flag, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		sp.status = CUDAFAIL;
		return sp;
	}
	for (i = 0; i < (p.n + 2); i++) sp.u[i] = p.alpha + i*(p.beta - p.alpha) / (p.n + 1);

	cudaStatus = cudaMemcpy(dev_u, sp.u, sizeof(float)*(p.n + 1), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		sp.status = CUDAFAIL;
		return sp;
	}
	cpuConvergenceTest = 0;
	k = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, solve_poison1d_kernel, 0, p.n-1);
	gridSize = (p.n-1 + blockSize - 1) / blockSize;
	for (i = 0; i < p.max_iter && !cpuConvergenceTest; i++) {
		cpuConvergenceTest = 1;

		cudaMemcpyToSymbol(flag, &cpuConvergenceTest, sizeof(int));

		solve_poison1d_kernel << <gridSize, blockSize >> >(h, dev_u, p.tol, p.a, p.b, p.n);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			sp.status = CUDAFAIL;
			return sp;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching solve_poison1_kernel!\n", cudaStatus);

			sp.status = CUDAFAIL;
			return sp;
		}
		cudaMemcpyFromSymbol(&cpuConvergenceTest, flag, sizeof(int));

		k++;
	}
	cudaMemcpy(sp.u, dev_u, sizeof(float)*(p.n + 1), cudaMemcpyDeviceToHost);

	*iter = k;
	if (k == p.max_iter) {
		sp.status = NOCONVERGENCE;
		return sp;
	}

	sp.status = CONVERGENCE;
	return sp;
}


float evaluate_solution(solution_poison1d s, float x, float a, float b, float n) {
	int i;
	i = (x - a)*(n + 1) / (b - a);
	return s.u[i];
}
int main()
{
	int iter = 0, i;
	problem_poison1d p;

	p.tol = powf(10, -5);
	p.max_iter = 10000;
	p.alpha = 0;
	p.beta = 1;
	p.a = 0;
	p.b = 1;
	p.n = 100;

	solution_poison1d s = solve_poison1d(p, &iter);

	switch (s.status) {
		case CONVERGENCE:
			printf("%d\n", iter);

			for (i = 0; i < p.n + 2; i++) {
				printf("u_%d=%f\n", i, s.u[i]);
			}
			break;
		case NOCONVERGENCE:
			printf("Method doesn't converge");
			break;
		case CUDAFAIL:
			printf("Something is wrong with CUDA");
			break;
	}

	if (_DEBUG) system("pause");
    return 0;
}

