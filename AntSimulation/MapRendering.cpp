using namespace std;
#include "Render.h"

#include <iostream>
#include <chrono>
#include <cstdlib>

void initArray(Map& map);
void generateMap(Map& map);
void fillMap(Map& map);
void smoothMap(Map& map);

void initMap(Map& map) {
	map.height = 100;
	map.width = 100;
	map.percentFill = 50;

	initArray(map);
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
	fillMap(map);
};

void fillMap(Map& map) {

	std::chrono::steady_clock::time_point t1 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < map.width; i++) {
		for (int j = 0; j < map.height; j++) {
			if (i == 0 || i == map.width - 1 || j == 0 || j == map.height - 1 || i == 1 || i == map.width - 2 || j == 1 || j == map.height - 2) {
				map.map[i][j] = 1; //solid outer wall
			}
			else {
				int randa = rand() % 100;
				map.map[i][j] = (randa < map.percentFill) ? 1 : 0; // 1 = solid, 0 = hollow
			}
		}
	}

	std::chrono::steady_clock::time_point t2 = std::chrono::high_resolution_clock::now();
	float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();

	cout << "   Fill Map:                 " << deltaTime << endl;
};
void smooth(Map& map);
void printMap(Map& map) {
	for (int i = 0; i < map.width; i++) {
		for (int j = 0; j < map.height; j++) {
			cout << map.map[i][j] << ",";
		}
		cout << "\n" << endl;
	}
};
void initArray(Map& map) {
	map.map = (int**)malloc(map.height * sizeof(int*));
	if (map.map) {

		for (int i = 0; i < map.width; i++) {
			map.map[i] = (int*)malloc(map.width * sizeof(int));

			for (int j = 0; j < map.width; j++) {
				map.map[i][j] = 0;
			}
		}
	}
};

