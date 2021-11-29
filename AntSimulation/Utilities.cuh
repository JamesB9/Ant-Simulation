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

////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#define _USE_MATH_DEFINES
#include <math.h>


////////////////////////////////////////////////////////////
/// \brief Defines a 2D vector (x, y) with useful operator overloads
/// 
/// \param x X-Coordinate of vector
/// \param y Y-Coordinate of vector
///
////////////////////////////////////////////////////////////
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

    __host__ __device__ Vec2f operator*(Vec2f a) {
        return { x * a.x, y * a.y };
    }

    __host__ __device__ Vec2f operator/(Vec2f a) {
        return { x / a.x, y / a.y };
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

    __host__ __device__ const Vec2f operator-(Vec2f a) const {
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

    __host__ __device__ float dotProduct(Vec2f b) {
        return this->x * b.x + this->y * b.y;
    }
};

////////////////////////////////////////////////////////////
/// CUDA BASED UTILITIES
////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////
/// \brief 
///
////////////////////////////////////////////////////////////
__device__ float cudaRand(curandState* state) {
	float randomf = curand_uniform(state);
	return randomf;
};

__device__ Vec2f randomInsideUnitCircle(curandState* state) {
	
	//a = distance from origin (0,0) from -1 to +1
	//r = angle from 0-360 degrees
	float a, r;
	a = (cudaRand(state) * 2.0f) - 1.0f;
	r = cudaRand(state) * 360.0f;


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

__device__ float getAngle(Vec2f a, Vec2f b) {
	float dot = a.x * b.x + a.y * b.y;
	float det = a.x * b.y - a.y * b.x;
	return atan2f(det, dot);
}

__device__ float getDistance(Vec2f a, Vec2f b) {
    return sqrtf(powf(b.x - a.x, 2.0f) + powf(b.y - a.y, 2.0f));
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

	return { dx, dy };
}