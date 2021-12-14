#include "ItemGrid.cuh"
#include "UnitTests.hpp"

namespace ItemGridUnitTests {

	// TEST 1: Intialising Item Grid on GPU
	TEST_RESULT test1() {
		// Setup
		ItemGrid* itemGrid = initItemGrid(100, 100);
		// Test
		bool isCreated = itemGrid->totalCells == 100 * 100;
		if (isCreated) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 2: Get Cell
	TEST_RESULT test2() {
		// Setup
		ItemGrid* itemGrid = initItemGrid(80, 80);
		itemGrid->worldCells[92].foodCount = 100;
		// Test
		Cell* cell = getCell(itemGrid, 125, 12);
		if (cell->foodCount == 100) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 3: Get Cell Index
	TEST_RESULT test3() {
		// Setup
		ItemGrid* itemGrid = initItemGrid(80, 80);

		// Test
		int cellIndex = getCellIndex(itemGrid, 125, 12);
		if (cellIndex == 92) return TEST_PASS;
		else return TEST_FAIL;
	}

	// TEST 4: Update Cell
	TEST_RESULT test4() {
		// Setup
		ItemGrid* itemGrid = initItemGrid(80, 80);
		itemGrid->worldCells[92].pheromones[0] = 50.0f;
		//printf("%f", itemGrid->worldCells[92].pheromones[0]);
		updateCell(itemGrid->worldCells[92], 1.0f);
		//printf("%f", itemGrid->worldCells[92].pheromones[0]);
			// Test
		bool hasDecayed = itemGrid->worldCells[92].pheromones[0] == 50.0f - Config::PHEROMONE_DECAY_STRENGH;
		if (hasDecayed) return TEST_PASS;
		else return TEST_FAIL;
	}
}