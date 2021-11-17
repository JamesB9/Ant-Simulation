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
#include "curand.h"
#include "curand_kernel.h"
#include "Utilities.cuh"

struct MoveComponent
{
    Vec2f position;
    Vec2f velocity, direction;
    float angle, maxSpeed, turningForce, roamStrength;
};

struct SniffComponent
{
    float sniffMaxDistance;
};

struct CollisionComponent {
    bool avoid; //Whether or not to follow targetPosition
    Vec2f targetPosition; //Position for ant to priorities
    float collisionDistance;
    Vec2f refractionPosition;
};

struct ActivityComponent
{
    int currentActivity;
};
