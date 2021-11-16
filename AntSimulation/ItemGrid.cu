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

int initItemGrid(ItemGrid& itemGrid, int worldX, int worldY) {
	itemGrid.worldX = worldX;
	itemGrid.worldY = worldY;
	itemGrid.totalCells = worldX * worldY;

	itemGrid.worldCells = createItemGridCellArray(itemGrid.totalCells);
	for (int i = 0; i < itemGrid.totalCells; i++) {
		itemGrid.worldCells[i].foodCount = 32;
		itemGrid.worldCells[i].pheromones[0] = 0.0f;
		itemGrid.worldCells[i].pheromones[1] = 0.0f;
	}
	return 0;
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
	return (floorf(y) * itemGrid.worldX) + floorf(x);
}