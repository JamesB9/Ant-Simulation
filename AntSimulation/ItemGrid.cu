#include "itemgrid.cuh"

Cell* createItemGridCellArray(int worldSize) {
	Cell* nArray;
	cudaMallocManaged(&nArray, worldSize * sizeof(Cell));
	return nArray;
}

int initItemGrid(ItemGrid& itemGrid, int worldX, int worldY) {
	itemGrid.worldCells = createItemGridCellArray(worldX * worldY);
	itemGrid.worldX = worldX;
	itemGrid.worldY = worldY;
	itemGrid.totalCells = worldX * worldY;

	return 0;
}

Cell* getCell(ItemGrid& itemGrid, float x, float y) {
	//Take X and Y, convert to 2D reference
	int posx = floorf(x);
	int posy = floorf(y);
	int index = posy * itemGrid.worldX;
	index += posx;


	return &itemGrid.worldCells[index];
}