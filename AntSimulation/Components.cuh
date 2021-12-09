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

////////////////////////////////////////////////////////////
/// Headers
////////////////////////////////////////////////////////////
#pragma once
#define LEAVING_HOME 0;
#define FOUND_FOOD 1;
#include <math.h>
#include "curand.h"
#include "curand_kernel.h"
#include "Utilities.cuh"


////////////////////////////////////////////////////////////
/// \brief ECS Entity Component to store data required for an entity to move
///
/// \param position 2D Vector storing the entity's location
/// \param velocity 2D Vector storing the entity's current velocity
/// \param direction 2D Vector storing the location the entity is facing
/// \param angle Rotation in degrees of the entity
/// \param maxSpeed The maximum distance to move per second
/// \param turningForce How tightly the entity turns away from obstacles
/// \param roamStrength How random the entity moves
///
////////////////////////////////////////////////////////////
struct MoveComponent
{
    Vec2f position;
    Vec2f velocity, direction;
    float angle, maxSpeed, turningForce, roamStrength;

    curandState state;
};


////////////////////////////////////////////////////////////
/// \brief ECS Entity Component to store data required for an entity to smell phermones
///
/// \param sniffMaxDistance The furthest distance a pheromone can be for an entity to detect it
/// \param sniffPheromone The type of pheromone the entity should be smelling (following)
///
////////////////////////////////////////////////////////////
struct SniffComponent
{
    float sniffMaxDistance;
    int sniffPheromone;
};


////////////////////////////////////////////////////////////
/// \brief ECS Entity Component to store data required for an entity to collide with map walls
///
/// \param avoid Whether or not to follow the targetPosition
/// \param targetPosition The position an entity moves towards to avoid walls
/// \param collisionDistance How close to a wall before an entity avoids it
/// \param refractionPosition Position calculated by reflecting angle of entity to wall
///
////////////////////////////////////////////////////////////
struct CollisionComponent {
    bool avoid;
    Vec2f targetPosition;
    float collisionDistance;
    Vec2f refractionPosition;
    bool stopPheromone;
};


////////////////////////////////////////////////////////////
/// \brief ECS Entity Component to store data required for an entity to know what it should do
///
/// \param currentActivity
/// \param maxDropStrength
/// \param dropStrength
/// \param dropStrengthReduction
/// \param timeSinceDrop
/// \param timePerDrop
/// \param lastFoodPickup
/// \param colonyId
///
////////////////////////////////////////////////////////////
struct ActivityComponent
{
    int currentActivity;
    float maxDropStrength;
    float dropStrength;
    float dropStrengthReduction;

    float timeSinceDrop;
    float timePerDrop;

    Vec2f lastFoodPickup;

    int colonyId;
};
