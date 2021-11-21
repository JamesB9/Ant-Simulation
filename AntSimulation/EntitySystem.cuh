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

MoveComponent* createMoveComponentArray(int n);
SniffComponent* createSniffComponentArray(int n);
ActivityComponent* createActivityComponentArray(int n);

Entities* initEntities(int entityCount);
int simulateEntitiesOnGPU(Entities* entities, ItemGrid* itemGrid, Map* map, float deltaTime);