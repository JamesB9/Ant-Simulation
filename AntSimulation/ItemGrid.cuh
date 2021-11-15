#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Components.cuh"
#include "stdio.h"
#include "math.h"

#define HOME_PHEROMONE 0;
#define FOOD_PHEROMONE 1;

struct Cell {
	float pheromones[2];
	float foodCount;
};

struct ItemGrid {
	int worldX;
	int worldY;
	int totalCells;
	Cell* worldCells;
};

int initItemGrid(ItemGrid& itemGrid, int worldX, int worldY);
Cell* getCell(ItemGrid& itemGrid, float x, float y);