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

////////////////////////////////////////////////////////////
/// Headers
////////////////////////////////////////////////////////////
#pragma once
#include "stdio.h"
#include "math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Config.hpp"
#define HOME_PHEROMONE 0;
#define FOOD_PHEROMONE 1;


////////////////////////////////////////////////////////////
/// \brief Stores data for each coordinate in an ItemGrid
///
/// \param pheromones array containing the intensity of each pheromone
/// \param foodCount The amount of food stored
///
////////////////////////////////////////////////////////////
struct Cell {
	float pheromones[2];
	float foodCount;
};


////////////////////////////////////////////////////////////
/// \brief A grid to hold pheromones and food
///
/// \param cellWidth Width of each Cell in world coordinates
/// \param cellHeight Height of each Cell in world coordinates
/// \param sizeX Number of Cells in each row of the grid
/// \param sizeY Number of Cells in each column of the grid
/// \param totalCells Number of cells in the grid (e.g. sizeX * sizeY)
/// \param worldCells 1D array of Cells
/// 
/// \see Cell
///
////////////////////////////////////////////////////////////
struct ItemGrid {
	float cellWidth;
	float cellHeight;
	int sizeX;
	int sizeY;
	int totalCells;
	Cell* worldCells;

	// HERE TEMPORARILY DUE TO CUDA NOT HAVING ACCESS TO CONFIG VARIABLES
	int worldX;
	int worldY;
};


////////////////////////////////////////////////////////////
/// \brief Creates an ItemGrid struct on the GPU in managed memory
///
/// \param sizeX Value to set as ItemGrid.sizeX
/// \param sizeY Value to set as ItemGrid.sizeY
/// 
/// \return pointer to an ItemGrid struct in managed memory
///
////////////////////////////////////////////////////////////
ItemGrid* initItemGrid(int sizeX, int sizeY);


////////////////////////////////////////////////////////////
/// \brief Returns Cell from 1D array ItemGrid.worldCells given 2D values x and y
///
/// \param itemGrid Reference to ItemGrid struct to act upon
/// \param x X-Coordinate of value in ItemGrid.worldCells
/// \param y Y-Coordinate of value in ItemGrid.worldCells
/// 
/// \return pointer to the cell from ItemGrid.worldCells at location (x,y)
///
////////////////////////////////////////////////////////////
Cell* getCell(ItemGrid& itemGrid, float x, float y);


////////////////////////////////////////////////////////////
/// \brief Returns 1D index given 2D values x and y
///
/// \param itemGrid Reference to ItemGrid struct to act upon
/// \param x X-Coordinate 
/// \param y Y-Coordinate
/// 
/// \return 1D index given 2D values (x,y)
///
////////////////////////////////////////////////////////////
int getCellIndex(ItemGrid& itemGrid, float x, float y);


////////////////////////////////////////////////////////////
/// \brief Updates a Cell by reducing its pheromone intensities
///
/// \param cell Reference to Cell struct to act upon
/// \param deltaTime The timestep between last update and now
///
////////////////////////////////////////////////////////////
void updateCell(Cell& cell, float deltaTime);


////////////////////////////////////////////////////////////
/// \brief [DEVICE FUNCTION] Returns Cell from 1D array ItemGrid.worldCells given 2D values x and y
///
/// \param itemGrid Reference to ItemGrid struct to act upon
/// \param x X-Coordinate of value in ItemGrid.worldCells
/// \param y Y-Coordinate of value in ItemGrid.worldCells
/// 
/// \return pointer to the cell from ItemGrid.worldCells at location (x,y)
///
////////////////////////////////////////////////////////////
__host__ __device__ Cell* getCell(ItemGrid* itemGrid, float mapx, float mapy);


////////////////////////////////////////////////////////////
/// \brief [DEVICE FUNCTION] Returns 1D index given 2D values x and y
///
/// \param itemGrid Reference to ItemGrid struct to act upon
/// \param x X-Coordinate 
/// \param y Y-Coordinate
/// 
/// \return 1D index given 2D values (x,y)
///
////////////////////////////////////////////////////////////
__host__ __device__ int getCellIndex(ItemGrid* itemGrid, float mapx, float mapy);

