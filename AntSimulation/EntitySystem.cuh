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