///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Components.cuh
// Description: The header file for all of the simulation componets.
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
#define LEAVING_HOME 0;
#define FOUND_FOOD 1;

#include <math.h>
#include <iostream>

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

    __host__ __device__ Vec2f operator+(Vec2f& a) {
        return { x + a.x, y + a.y };
    }

    __host__ __device__ Vec2f operator-(Vec2f& a) {
        return { x - a.x, y - a.y };
    }

    __host__ __device__ Vec2f operator+=(Vec2f& a) {
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

struct MoveComponent
{
    Vec2f position;
    Vec2f velocity, direction;
    float maxSpeed, turningForce;
};

struct SniffComponent
{
    float sniffMaxDistance;
};

struct ActivityComponent
{
    int currentActivity;
};
