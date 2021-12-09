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
#include <stdlib.h>
#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <queue>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Utilities.cuh"
#include "Config.hpp"
#include <SFML/System/Vector2.hpp>


////////////////////////////////////////////////////////////
/// \brief Defines a wall for entities to avoid
///
/// \param p1 Location of one end of the wall
/// \param p2 Location of second end of the wall
/// \param ID Identifier of wall
///
////////////////////////////////////////////////////////////
struct Boundary {
	Vec2f p1, p2;
	int ID;
};


////////////////////////////////////////////////////////////
/// \brief Holds information that make a map for entities to explore
///
/// \param height Number of Y Cells used to generate random map
/// \param width Number of X Cells used to generate random map
/// \param percentFill Percentage of the map cells that shouldn't be a wall
/// \param map array storing ints that define 0 for air and 1 for wall
/// \param walls array of boundaries that define the map outline
/// \param wallCount Number of boundaries in the boundary array
///
////////////////////////////////////////////////////////////
struct Map {
	int height;
	int width;
	int percentFill;

	int seed = 1111;
	int colonyQuadrant;

	int *map;
	Boundary* walls;
	int wallCount;
};


////////////////////////////////////////////////////////////
/// \brief 
///
/// \param x
/// \param y
///
////////////////////////////////////////////////////////////
struct Coord {
	int x;
	int y;
	Coord(int x, int y) {
		this->x = x;
		this->y = y;
	}
};


////////////////////////////////////////////////////////////
/// \brief Creates an empty map struct in GPU in managed memory
///
/// \param width Value to set as Map.width
/// \param height Value to set as Map.height
/// 
/// \return pointer to map struct in managed memory
///
////////////////////////////////////////////////////////////
Map* makeMapPointer(int width, int height);


////////////////////////////////////////////////////////////
/// \brief Creates a map struct from image in GPU in managed memory
///
/// \param path File path to the image to load
/// 
/// \return pointer to map struct in managed memory
///
////////////////////////////////////////////////////////////
Map* makeMapPointer(std::string path);


////////////////////////////////////////////////////////////
/// \brief Returns value from 1D array Map.map given 2D values x and y
///
/// \param map Reference to Map struct to act upon
/// \param x X-Coordinate of value in Map.map
/// \param y Y-Coordinate of value in Map.map
/// 
/// \return value from Map.map at location (x,y)
///
////////////////////////////////////////////////////////////
int getMapValueAt(Map& map, int x, int y);


////////////////////////////////////////////////////////////
/// \brief Sets the value in 1D array Map.map given 2D values x and y
///
/// \param map Reference to Map struct to act upon
/// \param x X-Coordinate of value to set in Map.map
/// \param y Y-Coordinate of value to set Map.map
/// 
////////////////////////////////////////////////////////////
void setMapValueAt(Map& map, int x, int y, int val);
bool isValidForColonyAndFood(Map& map, int x, int y);
void initBlankMap(Map* map, int height, int width);
void generateMap(Map& map);
void fillMap(Map& map);
void smoothMap(Map& map);
void floodFill(Map& map);
void arrayCopy(Map& from, Map& to);
int getNeighbourWallCount(Map& map, int x, int y, int delta);
void createMap(Map* map);
void printMap(Map& map);
void initArray(Map* map);
int getNumAreas(Map& map);
sf::Vector2i* getValidArea(Map& map, int quadrant);
sf::Vector2i* foodLocation(Map& map);
sf::Vector2i* colonyLocation(Map& map);