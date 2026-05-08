#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

double *d_mass;

void compute(){

    cudaMalloc(&d_hPos, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&d_hPos, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&mass, sizeof(double)*NUMENTITIES);

    cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass,mass,sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);


    int threads = 256;
    int blocks = (NUMENTITIES+threads-1) / threads;


    //kernel
    compute_kernel<<<blocks,threads>>>(d_hPos,d_hVel,d_mass);


    cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_mass,mass,sizeof(double)*NUMENTITIES, cudaMemcpyDeviceToHost);

    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}

__global__ void compute_kernel(vector3 *d_hPos,vector3 *d_hVel, double *mass){

}