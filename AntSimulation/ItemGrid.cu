///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Utilities.cu
// Description: The item grid implmentation for the simulation.
// 
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:  
//	- (URL)
// 
// 
//////////////////////////// 80 columns wide //////////////////////////////////
#include "itemgrid.cuh"

Cell* createItemGridCellArray(int worldSize) {
	Cell* nArray;
	cudaMallocManaged(&nArray, worldSize * sizeof(Cell));
	return nArray;
}

ItemGrid* initItemGrid(int sizeX, int sizeY) {
	ItemGrid* itemGrid;
	cudaMallocManaged(&itemGrid, sizeof(ItemGrid));

	itemGrid->sizeX = sizeX;
	itemGrid->sizeY = sizeY;
	itemGrid->totalCells = sizeX * sizeY;

	// TEMP WORLD VARS
	itemGrid->worldX = Config::WORLD_SIZE_X;
	itemGrid->worldY = Config::WORLD_SIZE_Y;

	itemGrid->worldCells = createItemGridCellArray(itemGrid->totalCells);
	for (int i = 0; i < itemGrid->totalCells; i++) {
		itemGrid->worldCells[i].foodCount = 0.0f;
		itemGrid->worldCells[i].pheromones[0] = 0.0f;
		itemGrid->worldCells[i].pheromones[1] = 0.0f;

		itemGrid->worldCells[i].timePerDrop = Config::PHEROMONE_DECAY_TIME;
		itemGrid->worldCells[i].timeSinceDrop = 0.0f;
	}
	return itemGrid;
}

Cell* getCell(ItemGrid& itemGrid, float x, float y) {
	//Take X and Y, convert to 2D reference
	//int posx = floorf(x);
	//int posy = floorf(y);
	//int index = posy * itemGrid.worldX;
	//index += posx;


	return &itemGrid.worldCells[getCellIndex(itemGrid, x, y)];
}

int getCellIndex(ItemGrid& itemGrid, float x, float y) {
	return (floorf(y) * itemGrid.sizeX) + floorf(x);
}

void updateCell(Cell& cell, float deltaTime) {
	if (cell.pheromones[0] > 0.0f || cell.pheromones[1] > 0.0f) {
		cell.timeSinceDrop += deltaTime;
		if (cell.timeSinceDrop > cell.timePerDrop) {
			cell.pheromones[0] > Config::PHEROMONE_DECAY_STRENGH ? cell.pheromones[0] -= Config::PHEROMONE_DECAY_STRENGH : cell.pheromones[0] = 0.0f;
			cell.pheromones[1] > Config::PHEROMONE_DECAY_STRENGH ? cell.pheromones[1] -= Config::PHEROMONE_DECAY_STRENGH : cell.pheromones[1] = 0.0f;

			cell.timeSinceDrop = 0.0f;
		}
	}
}


int getCellIndex(ItemGrid* itemGrid, float mapx, float mapy) {
	float widthOfCell = Config::MAP_SIZE_X / itemGrid->sizeX;
	float heightOfCell = Config::MAP_SIZE_Y / itemGrid->sizeY;
	return (floorf(mapy / heightOfCell) * itemGrid->sizeX) + floorf(mapx / widthOfCell);
}

Cell* getCell(ItemGrid* itemGrid, float mapx, float mapy) {
	return &itemGrid->worldCells[getCellIndex(itemGrid, mapx, mapy)];
}