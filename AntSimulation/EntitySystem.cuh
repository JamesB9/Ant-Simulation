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
#include "math.h"
#include "Entities.cuh"

MoveComponent* createMoveComponentArray(int n);
SniffComponent* createSniffComponentArray(int n);

int initEntities(Entities& entities);
int simulateEntitiesOnGPU(Entities& entities, float deltaTime);