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

	return { 0.0f + (a * cos(r)), 0.0f + (a * sin(r)) };
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

__device__ float getAngle(Vec2f a, Vec2f b) {
	float dot = a.x * b.x + a.y * b.y;
	float det = a.x * b.y - a.y * b.x;
	return atan2f(det, dot);
}

__device__ bool isLeft(Vec2f a, Vec2f b, Vec2f c) {
	return ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)) > 0;
}

__device__ float normaliseRadian(float a) {
	a = fmodf(a, M_PI);
	if (a < 0) { a += M_PI; }
	return a;
}

__device__ Vec2f normaliseSurface(Vec2f a, Vec2f b) {
	float dx = b.x - a.x;
	float dy = b.y - a.y;

	return { dx, -dy };
}