///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Utilities.cuh
// Description: Utiltiy functions for cuda functions.
// 
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:  
//	- (URL)
// 
// 
//////////////////////////// 80 columns wide //////////////////////////////////
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "Components.cuh"

//CUDA Based Utilities

__device__ float cudaRand(unsigned int seed, float start, float end) {
	//int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curandState_t state;
	curand_init(seed,
		blockIdx.x,
		0,
		&state
	);
	//Cuda random function
	float randomf = curand_uniform(&state);
	randomf *= (end - start + 0.999999);
	randomf += start;

	return randomf;
};

__device__ Vec2f randomInsideUnitCircle(unsigned int seed) {
	Vec2f out;
	out.x = cudaRand(seed, -1.0f, 1.0f);
	out.y = cudaRand(seed, -1.0f, 1.0f);
	return out;
}

__device__ Vec2f clamp(Vec2f v, float max) {
	if (fabs(v.x) > max || fabs(v.y) > max) { // x or y larger than desired max
		if (fabs(v.x) > fabs(v.y)) { //x bigger than y?
			v = v * (max / fabs(v.x)); // scale whole vector by factor of max/x
		}
		else {
			v = v * (max / fabs(v.y)); // scale whole vector by factor of max/y
		}
	}

	return v;
}