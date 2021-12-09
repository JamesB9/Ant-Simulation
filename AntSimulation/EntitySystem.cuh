///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: EntitySystem.cuh
// Description: The headder for the entity sytstem.
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
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#define _USE_MATH_DEFINES
#include "math.h"
#include "Entities.cuh"
#include "ItemGrid.cuh"
#include "Map.cuh"
#include "Config.hpp"
#include "Colony.cuh"


////////////////////////////////////////////////////////////
/// \brief Creates an array of move components on the GPU in managed memory
///
/// \param n Number of move comonents (size of array)
/// 
/// \return pointer to an array of move components in managed memory
///
////////////////////////////////////////////////////////////
MoveComponent* createMoveComponentArray(int n);


////////////////////////////////////////////////////////////
/// \brief Creates an array of sniff components on the GPU in managed memory
///
/// \param n Number of sniff comonents (size of array)
/// 
/// \return pointer to an array of sniff components in managed memory
///
////////////////////////////////////////////////////////////
SniffComponent* createSniffComponentArray(int n);


////////////////////////////////////////////////////////////
/// \brief Creates an array of activity components on the GPU in managed memory
///
/// \param n Number of activity comonents (size of array)
/// 
/// \return pointer to an array of activity components in managed memory
///
////////////////////////////////////////////////////////////
ActivityComponent* createActivityComponentArray(int n);


////////////////////////////////////////////////////////////
/// \brief Creates an Entities struct on the GPU in managed memory
///
/// \param colonies Array of Colony structs used to create entities
/// \param entityCount Number of entities to create
/// 
/// \return pointer to an Entities struct in managed memory
///
////////////////////////////////////////////////////////////
Entities* initEntities(Colony* colonies, int entityCount);


////////////////////////////////////////////////////////////
/// \brief Wrapper function to launch CUDA kernel function simulateEntities
///
/// \param entities Entities storing data to perform simulation upon
/// \param itemGrid ItemGrid storing pheromone data used for simulation
/// \param map Map storing walls for entities to avoid for simulation
/// \param colonies Colonies used for entities for simulation
/// \param deltaTime Timestep in simulation between last simulation and now
/// 
/// \see simulateEntities()
/// 
////////////////////////////////////////////////////////////
int simulateEntitiesOnGPU(Entities* entities, ItemGrid* itemGrid, Map* map, Colony* colonies, float deltaTime);
void setupStatesOnGPU(Entities* entities);