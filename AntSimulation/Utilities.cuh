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


struct Vec2f {
    float x, y;

    __host__ __device__ Vec2f operator*=(float& a) {
        this->x *= a;
        this->y *= a;

        return *this;
    }

    __host__ __device__ Vec2f operator*(float a) {
        return { x * a, y * a };
    }

    __host__ __device__ Vec2f operator/(float a) {
        return { x / a, y / a };
    }

    __host__ __device__ Vec2f operator+(Vec2f a) {
        return { x + a.x, y + a.y };
    }

    __host__ __device__ Vec2f operator-(Vec2f a) {
        return { x - a.x, y - a.y };
    }

    __host__ __device__ Vec2f operator+=(Vec2f a) {
        this->x += a.x;
        this->y += a.y;

        return *this;
    }

    __host__ __device__ Vec2f operator=(float angle) {
        this->x = cos(angle);
        this->y = sin(angle);

        return *this;
    }
};

//CUDA Based Utilities

__device__ float cudaRand(curandState* state) {
	float randomf = curand_uniform(state);
	return randomf;
};

__device__ Vec2f randomInsideUnitCircle(curandState* state) {
	
	//a = distance from origin (0,0) from -1 to +1
	//r = angle from 0-360 degrees
	float a, r;
	a = (cudaRand(state) * 2.0f) - 1.0f;
	//r = cudaRand(state) * 360.0f;
	r = (cudaRand(state) * 360.0f);

	return { 0.0f + (a * (float)cos(r)), 0.0f + (a * (float)sin(r)) };
	//return { a, r };
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