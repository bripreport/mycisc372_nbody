#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vector.h"
#include "config.h"

#define HANDLE_ERROR(call){ \
    cudaError_t err = (call); \
    if(err != cudaSuccess){ \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1);\
    }\
}


double *d_mass;


__global__ void compute_kernel(vector3 *d_hPos,vector3 *d_hVel, double *d_mass){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i < NUMENTITIES){
        vector3 accels = {0,0,0};
        for(int j = 0; j < NUMENTITIES; j++){
            if(i == j){
                continue;
            }
            else{
                vector3 distance;

                for (int k=0;k<3;k++) {
                    distance[k]=d_hPos[i][k] - d_hPos[j][k];
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
		}

    }
}

__global__ void update_pos_kernel(vector3 *d_hPos, vector3 *d_hVel){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < NUMENTITIES){
        for (int k=0;k<3;k++){
			d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
		}
    }
}



extern "C" void compute(){

    HANDLE_ERROR( cudaMalloc(&d_hPos, sizeof(vector3)*NUMENTITIES) );
    HANDLE_ERROR(cudaMalloc(&d_hVel, sizeof(vector3)*NUMENTITIES));
    HANDLE_ERROR(cudaMalloc(&d_mass, sizeof(double)*NUMENTITIES));

    HANDLE_ERROR(cudaMemcpy(d_hPos,hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_hVel,hVel,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_mass,mass,sizeof(double)*NUMENTITIES, cudaMemcpyHostToDevice));


    int threads = 256;
    int blocks = (NUMENTITIES+threads-1) / threads;

    //kernel
    compute_kernel<<<blocks,threads>>>(d_hPos,d_hVel,d_mass);
    HANDLE_ERROR(cudaDeviceSynchronize());

    update_pos_kernel<<<blocks,threads>>>(d_hPos,d_hVel);
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(hPos,d_hPos,sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(hVel,d_hVel,sizeof(vector3)*NUMENTITIES, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(mass,d_mass,sizeof(double)*NUMENTITIES, cudaMemcpyDeviceToHost));

    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
}


/*
 * ChatGPT was used to help with final debugging, explaining existing code/functions, and correcting errors that showed up when compiling. 
 * It was also used to help understand some C and CUDA-specific logic and address mistakes I made with syntax and pointers. 
 * It helped with understanding the math of the computations and making sure I'm correctly using the global device and host variables properly.
 * It also helped with making an error handling macro similar to ones seen in the notes, at an attempt to potentially catch issues I was facing where the position values would be heavily inconsistent.
 */
