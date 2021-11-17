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
using namespace std;
#include "Map.cuh"

#include <string>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <queue>

void initArray(Map& map);
void initBlankMap(Map& map, int height, int width);
void generateMap(Map& map);
void fillMap(Map& map);
void smoothMap(Map& map);
void floodFill(Map& map);
void arrayCopy(Map& from, Map& to);
int getNeighbourWallCount(Map& map, int x, int y, int delta);

//need to define in .h file

void initMap(Map& map) {
	map.height = 100;
	map.width = 100;
	map.percentFill = 50;

	initArray(map);
	generateMap(map);
	printMap(map);
}
void initMap(Map& map,int height, int width) {
	map.height = height;
	map.width = width;
	map.percentFill = 48;

	initArray(map);
	generateMap(map);
	printMap(map);
}


void generateMap(Map& map) {

	bool enableTiming = true;
	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	float deltaTime;

	if(enableTiming)
		t1 = std::chrono::high_resolution_clock::now();
	
	fillMap(map);

	if (enableTiming) {
		t2 = std::chrono::high_resolution_clock::now();
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
		cout << "   Fill Map:                 " << deltaTime << endl;
	}

	for (int i = 0; i < 8; i++) {
		if(enableTiming)
			t1 = std::chrono::high_resolution_clock::now();
		smoothMap(map);
		if (enableTiming) {
			t2 = std::chrono::high_resolution_clock::now();
			deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
			cout << "   Smooth Map (" << i << "):           " << deltaTime << endl;
		}
	}
	floodFill(map);
	printMap(map);

};

void fillMap(Map& map) {
	srand(1111);
	
	for (int i = 0; i < map.width; i++) {
		for (int j = 0; j < map.height; j++) {
			if (i == 0 || i == map.width - 1 || j == 0 || j == map.height - 1 || i == 1 || i == map.width - 2 || j == 1 || j == map.height - 2) {
				setMapValueAt(map, i, j, 1); //solid outer wall
			}
			else {
				int randa = rand() % 100;
				(randa < map.percentFill) ? setMapValueAt(map, i, j, 1) : setMapValueAt(map, i, j, 0); // 1 = solid, 0 = hollow
			}
		}
	}

	
};
void smoothMap(Map& map) {
	for (int i = 0; i < map.width; i++) {
		for (int j = 0; j < map.height; j++) {
			int wallCount = getNeighbourWallCount(map, i, j, 1);
			if (wallCount > 4) {
				setMapValueAt(map, i, j, 1);
			}
			else if (wallCount < 4) {
				setMapValueAt(map, i, j, 0);
			}
		}
	}
};
void floodFill(Map& map) {
	Map visited;

	initBlankMap(visited, map.height, map.height);
	arrayCopy(map, visited);

	for (int i = 0; i < map.width; i++) {
		for (int j = 0; j < map.height; j++) {
			if (getMapValueAt(visited, i, j) == 0) {
				cout << i << "," << j << endl;
				Coord coord = Coord(i, j);
				std::queue<Coord> coordQueue;
				std::queue<Coord> visitedQueue;

				coordQueue.push(coord);
				visitedQueue.push(coord);
				while (coordQueue.size() != 0) {
					Coord coord = coordQueue.front();
					coordQueue.pop();
					int x = coord.x;
					int y = coord.y;

					if ((x - 1 >= 0) && getMapValueAt(visited, x-1, y) == 0) {
						//cout << "add left" << endl;
						coordQueue.push(Coord(x - 1, y));
						visitedQueue.push(Coord(x - 1, y));
						setMapValueAt(visited, x-1,y, 1);
					}
					if ((x + 1 < map.width) && getMapValueAt(visited, x + 1, y) == 0) {
						//cout << "add right" << endl;
						coordQueue.push(Coord(x + 1, y));
						visitedQueue.push(Coord(x + 1, y));
						setMapValueAt(visited, x + 1, y, 1);
					}
					if ((y - 1 >= 0) && getMapValueAt(visited, x, y - 1) == 0) {
						//cout << "add up" << endl;
						coordQueue.push(Coord(x, y - 1));
						visitedQueue.push(Coord(x, y - 1));
						setMapValueAt(visited, x, y - 1, 1);
					}
					if ((y + 1 < map.height) && getMapValueAt(visited, x, y + 1) == 0) {
						//cout << "add down" << endl;
						coordQueue.push(Coord(x, y + 1));
						visitedQueue.push(Coord(x, y + 1));
						setMapValueAt(visited, x, y + 1, 1);
					}
				}
				cout << visitedQueue.size() << endl;
				if (visitedQueue.size() <= 80) {//remove all areas smaller than 20
					while (visitedQueue.size() != 0) {
						Coord coord = visitedQueue.front();
						visitedQueue.pop();
						setMapValueAt(map, coord.x, coord.y, 1);
					}
				}
			}
		}
	}
}


//Util Functions
void arrayCopy(Map& from, Map& to) {
	for (int i = 0; i < from.height * from.width; i++) {
		to.map[i] = from.map[i];
	}
}

void printMap(Map& map) {
	for (int i = 0; i < map.width; i++) {
		for (int j = 0; j < map.height; j++) {
			cout << getMapValueAt(map, i, j) << ",";
		}
		cout << "\n" << endl;
	}
};
int getNeighbourWallCount(Map& map, int x, int y, int delta) {

	if (delta <= 0) return 0;
	//cout << x << "," << y << endl;
	int wallCount = 0;
	for (int nX = x - delta; nX <= x + delta; nX++) {
		for (int nY = y - delta; nY <= y + delta; nY++) {
			if (nX >= 0 && nX < map.width && nY >= 0 && nY < map.height) {
				if (getMapValueAt(map,nX,nY) == 1) {
					//cout << "|" << nX << "," << nY;
					if (!(nX == x && nY == y)) {
						wallCount++;
					}
				}
			}
		}
	}
	//cout << wallCount << endl;
	return wallCount;
}
void initArray(Map& map) {
	map.map = (int*)malloc(map.height * map.width * sizeof(int));
	if (map.map) {
		for (int i = 0; i < (map.height * map.width); i++) {
			map.map[i] = 0;
		}
		
	}
};
void initBlankMap(Map& map, int height, int width) {
	map.height = height;
	map.width = width;
	map.percentFill = 0;
	initArray(map);
}

