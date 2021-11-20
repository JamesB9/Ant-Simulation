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

#include "stdio.h"
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define HOME_PHEROMONE 0;
#define FOOD_PHEROMONE 1;

struct Cell {
	float pheromones[2];
	float foodCount;

	float timeSinceDrop;
	float timePerDrop;
};

struct ItemGrid {
	int worldX;
	int worldY;
	int totalCells;
	Cell* worldCells;
};

ItemGrid* initItemGrid(int worldX, int worldY);
Cell* getCell(ItemGrid& itemGrid, float x, float y);
int getCellIndex(ItemGrid& itemGrid, float x, float y);

void updateCell(Cell& cell, float deltaTime);

Cell* getCell(ItemGrid* itemGrid, float mapx, float mapy);
int getCellIndex(ItemGrid* itemGrid, float mapx, float mapy);