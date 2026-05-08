#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

double *d_mass;


__global__ void compute_kernel(vector3 *d_hPos,vector3 *d_hVel, double *d_mass){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < NUMENTITIES){
        double *i_pos = d_hPos[i];

        vector3 accels = {0,0,0};
        for(int j = 0; j < NUMENTITIES; j++){
            double *j_pos = d_hPos[j];
            if(i == j){
                continue;
            }
            else{
                vector3 distance;

                for (int k=0;k<3;k++) {
                    distance[k]=i_pos[k] - j_pos[k];
                }
                
                double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;

                for(int k = 0; k < 3; k++){
                    accels[k] += accelmag*distance[k]/magnitude;
                }
            }
        }
        for (int k=0;k<3;k++){
			d_hVel[i][k]+=accels[k]*INTERVAL;
			d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
		}




    }
}



extern "C" void compute(){

    cudaMalloc(&d_hPos, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&d_hVel, sizeof(vector3)*NUMENTITIES);
    cudaMalloc(&d_mass, sizeof(double)*NUMENTITIES);

    cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel,hVel,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass,mass,sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice);


    int threads = 256;
    int blocks = (NUMENTITIES+threads-1) / threads;

    //kernel
    compute_kernel<<<blocks,threads>>>(d_hPos,d_hVel,d_mass);
    cudaDeviceSynchronize();


    cudaMemcpy(hPos,d_hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel,d_hVel,sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(mass,d_mass,sizeof(double)*NUMENTITIES, cudaMemcpyDeviceToHost);

    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}