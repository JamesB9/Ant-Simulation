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

Map* makeMapPointer(int width, int height) {
	Map* map;
	cudaMallocManaged(&map, sizeof(Map));

	map->width = width;
	map->height = height;
	int* intMap;
	cudaMallocManaged(&intMap, sizeof(int) * width * height);
	map->map = intMap;
	return map;
};

void createMap(Map* map, int width, int height);

void printMap(Map& map);

static int getMapValueAt(Map& map, int x, int y) {
	return map.map[(y * map.width) + x];
};
static void setMapValueAt(Map& map, int x, int y, int val) {
	map.map[(y * map.width) + x] = val;
}