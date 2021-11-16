///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Utilities.cuh
// Description: The item grid header file for the simulation.
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
int getCellIndex(ItemGrid& itemGrid, float x, float y);