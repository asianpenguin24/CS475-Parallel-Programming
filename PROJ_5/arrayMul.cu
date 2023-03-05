//Jeremy Udarbe
//CS 475
//Project 5 - CUDA Monte Carlo
#define _USE_MATH_DEFINES
// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE 128 // number of threads in each block
#endif
#ifndef NUMTRIALS // WARNING: DON’T CALL THIS “ARRAYSIZE” !
#define NUMTRIALS ( 8*1024*1024 ) // size of the array
#endif
float hA[DATASET_SIZE];
float hB[DATASET_SIZE];
float hC[DATASET_SIZE];

void    CudaCheckError();
void	TimeOfDaySeed();
float Ranf(float low, float high);
int Ranf(int ilow, int ihigh);

//global variables
const float GRAVITY = -9.8;	// acceleraion due to gravity in meters / sec^2

// degrees-to-radians -- callable from the device:
__device__ float Radians(float d) {
    return (M_PI / 180.f) * d;
}

// the kernel:
__global__ void MonteCarlo(float* dvs, float* dths, float* dgs, float* dhs, float* dds, int* dhits) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // randomize everything:
    float v = dvs[gid];
    float thr = Radians(dths[gid]);
    float vx = v * cos(thr);
    float vy = v * sin(thr);
    float  g = dgs[gid];
    float  h = dhs[gid];
    float  d = dds[gid];

    int numHits = 0;

    // see if the ball doesn't even reach the cliff:
    float t = -vy / (0.5 * GRAVITY);
    float x = vx * t;
    if (x > g) {
        ...
            numHits = 1;
    }

    dhits[gid] = numHits;
}


// these two #defines are just to label things
// other than that, they do nothing:
#define IN
#define OUT

int main(int argc, char* argv[]) {
    TimeOfDaySeed();

    int dev = findCudaDevice(argc, (const char**)argv);

    // better to define these here so that the rand() calls don't get into the thread timing:
    float* hvs = new float[NUMTRIALS];
    float* hths = new float[NUMTRIALS];
    float* hgs = new float[NUMTRIALS];
    float* hhs = new float[NUMTRIALS];
    float* hds = new float[NUMTRIALS];
    int* hhits = new int[NUMTRIALS];

    // fill the random-value arrays:

    ? ? ? ? ?


        // allocate device memory:
        float* dvs, * dths, * dgs, * dhs, * dds;
    int* dhits;

    cudaMalloc(&dvs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dths, NUMTRIALS * sizeof(float));
    cudaMalloc(&dgs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dhs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dds, NUMTRIALS * sizeof(float));
    cudaMalloc(&dhits, NUMTRIALS * sizeof(int));
    CudaCheckError();

    // copy host memory to the device:
    cudaMemcpy(dvs, hvs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dths, hths, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dgs, hgs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dhs, hhs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dds, hds, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    CudaCheckError();

    // setup the execution parameters:
    dim3 grid(NUMBLOCKS, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // allocate cuda events that we'll use for timing:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CudaCheckError();

    // let the gpu go quiet:
    cudaDeviceSynchronize();

    // record the start event:
    cudaEventRecord(start, NULL);
    CudaCheckError();

    // execute the kernel:
    MonteCarlo << < grid, threads >> > (IN dvs, IN dths, IN dgs, IN dhs, IN dds, OUT dhits);

    // record the stop event:
    cudaEventRecord(stop, NULL);
    CudaCheckError();

    // wait for the stop event to complete:
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    CudaCheckError();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    CudaCheckError();

    // compute and print the performance

    ? ? ? ? ?

        // copy result from the device to the host:
        cudaMemcpy(hhits, dhits, NUMTRIALS * sizeof(int), cudaMemcpyDeviceToHost);
    CudaCheckError();

    // add up the hhits[ ] array: :

    ? ? ? ? ?

        // compute and print the probability:

        ? ? ? ? ?

        // clean up host memory:
        delete[] hvs;
    delete[] hths;
    delete[] hgs;
    delete[] hhs;
    delete[] hds;
    delete[] hhits;

    // clean up device memory:
    cudaFree(dvs);
    cudaFree(dths);
    cudaFree(dgs);
    cudaFree(dhs);
    cudaFree(dds);
    cudaFree(dhits);
    CudaCheckError();

    return 0;
}

void CudaCheckError() {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    }
}

void TimeOfDaySeed() {
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

    time_t  timer;
    time(&timer);
    double seconds = difftime(timer, mktime(&y2k));
    unsigned int seed = (unsigned int)(1000. * seconds);    // milliseconds
    srand(seed);
}

float Ranf(float low, float high) {
    float r = (float)rand();               // 0 - RAND_MAX
    float t = r / (float)RAND_MAX;       // 0. - 1.

    return   low + t * (high - low);
}

int Ranf(int ilow, int ihigh) {
    float low = (float)ilow;
    float high = ceil((float)ihigh);

    return (int)Ranf(low, high);
}
