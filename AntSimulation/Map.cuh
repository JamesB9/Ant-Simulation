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

//Pool all render.cpp's here

struct Boundary {
	Vec2f p1, p2;
	int ID;
};

struct Map {
	int height;
	int width;
	int percentFill;

	int *map;
	Boundary* walls;
	int wallCount;
};

struct Coord {
	int x;
	int y;
	Coord(int x, int y) {
		this->x = x;
		this->y = y;
	}
};

Map* makeMapPointer(int width, int height);
Map* makeMapPointer(std::string path);
int getMapValueAt(Map& map, int x, int y);
void setMapValueAt(Map& map, int x, int y, int val);

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