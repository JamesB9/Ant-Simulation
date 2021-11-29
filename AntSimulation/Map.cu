#include "Map.cuh"
#include <string>
#include <algorithm>

#include <SFML/Graphics/Image.hpp>

void initMap(Map* map) {
	for (int i = 0; i < (map->height * map->width); i++) {
		map->map[i] = 0;
	}
}

Map* makeMapPointer(std::string path) {
	Map* map;

	cudaMallocManaged(&map, sizeof(Map));

	sf::Image imgMap;
	if (!imgMap.loadFromFile(path)) {
		std::cout << "ERROR" << std::endl;

	}
	else {
		std::cout << "Succesfully loaded map: " << path << std::endl;
	}

	map->width = imgMap.getSize().x;
	map->height = imgMap.getSize().y;

	int* intMap;
	cudaMallocManaged(&intMap, sizeof(int) * map->width * map->height);
	map->map = intMap;

	for (int i = 0; i < map->width; i++) {
		for (int j = 0; j < map->height; j++) {
			sf::Color color = imgMap.getPixel(i, j);
			if (color == sf::Color::Black) {
				setMapValueAt(*map, i, j, 1);
			}
			else if(color == sf::Color::White) {
				setMapValueAt(*map, i, j, 0);
			}
		}
	}

	return map;
}

Map* makeMapPointer(int width, int height) {
	Map* map;
	cudaMallocManaged(&map, sizeof(Map));
	map->width = width;
	map->height = height;
	map->seed = (Config::MAP_SEED == -1) ? time(NULL) : Config::MAP_SEED;
	int* intMap;
	cudaMallocManaged(&intMap, sizeof(int) * width * height);
	map->map = intMap;
	return map;
};


int getMapValueAt(Map& map, int x, int y) {
	return map.map[(y * map.width) + x];
};
void setMapValueAt(Map& map, int x, int y, int val) {
	map.map[(y * map.width) + x] = val;
}

//Timing Data
bool enableTiming = true;


void createMap(Map* map) {
	map->percentFill = 48;

	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	float deltaTime;
	if (enableTiming)
		std::cout << "\nMAP GENERATION" << std::endl;

	//initArray(map);
	if (enableTiming)
		t1 = std::chrono::high_resolution_clock::now();
	generateMap(*map);
	//printf("HELLO");
	if (enableTiming) {
		t2 = std::chrono::high_resolution_clock::now();
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
		std::cout << "\n   Time to fully generate Map: " << deltaTime << "\n" << std::endl;
	}
	//printMap(map);
}


void generateMap(Map& map) {


	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	float deltaTime;

	if (enableTiming)
		t1 = std::chrono::high_resolution_clock::now();

	fillMap(map);

	if (enableTiming) {
		t2 = std::chrono::high_resolution_clock::now();
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
		std::cout << "   Fill Map:                 " << deltaTime << std::endl;
	}

	for (int i = 0; i < 8; i++) {
		if (enableTiming)
			t1 = std::chrono::high_resolution_clock::now();
		smoothMap(map);
		if (enableTiming) {
			t2 = std::chrono::high_resolution_clock::now();
			deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
			std::cout << "   Smooth Map (" << i << "):           " << deltaTime << std::endl;
		}
	}
	if (enableTiming)
		t1 = std::chrono::high_resolution_clock::now();
	floodFill(map);
	if (enableTiming) {
		t2 = std::chrono::high_resolution_clock::now();
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
		std::cout << "   Flood Fill:               " << deltaTime << std::endl;
	}
};

void fillMap(Map& map) {
	srand(map.seed);

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

	initBlankMap(&visited, map.height, map.height);
	arrayCopy(map, visited);

	for (int i = 0; i < map.width; i++) {
		for (int j = 0; j < map.height; j++) {
			if (getMapValueAt(visited, i, j) == 0) {


				//cout << i << "," << j << endl;


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

					if ((x - 1 >= 0) && getMapValueAt(visited, x - 1, y) == 0) {
						//cout << "add left" << endl;
						coordQueue.push(Coord(x - 1, y));
						visitedQueue.push(Coord(x - 1, y));
						setMapValueAt(visited, x - 1, y, 1);
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


				//cout << visitedQueue.size() << endl;


				if (visitedQueue.size() <= map.width) {//remove all areas smaller than 20
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
			std::cout << getMapValueAt(map, i, j) << ",";
		}
		std::cout << "\n" << std::endl;
	}
};
int getNeighbourWallCount(Map& map, int x, int y, int delta) {

	if (delta <= 0) return 0;
	//cout << x << "," << y << endl;
	int wallCount = 0;
	for (int nX = x - delta; nX <= x + delta; nX++) {
		for (int nY = y - delta; nY <= y + delta; nY++) {
			if (nX >= 0 && nX < map.width && nY >= 0 && nY < map.height) {
				if (getMapValueAt(map, nX, nY) == 1) {
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

void initArray(Map* map) {
	map->map = (int*)malloc(map->height * map->width * sizeof(int));
	if (map->map) {
		for (int i = 0; i < (map->height * map->width); i++) {
			//printf("%d\n", i);
			map->map[i] = 0;
		}

	}
};
void initBlankMap(Map* map, int height, int width) {
	map->height = height;
	map->width = width;
	map->percentFill = 0;
	initArray(map);
}
sf::Vector2i* foodLocation(Map& map) {
	srand(map.seed);
	int randa = (rand() % 4);
	srand(time(NULL));
	randa = (randa + ((rand() % 3) + 1)) % 4;//Randomly pick between 3 other quadrants colony is not in

	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	float deltaTime;

	std::cout << "Food Quadrant: " << randa << std::endl;

	if (enableTiming)
		t1 = std::chrono::high_resolution_clock::now();
	std::cout << "-----FOOD-----" << std::endl;

	sf::Vector2i* foodPos = getValidArea(map, randa); //randa: 0 = TL, 1 = TR, 2 = BR, 3 = BL

	if (enableTiming) {
		t2 = std::chrono::high_resolution_clock::now();
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
		std::cout << "      Valid Food Pos Time: " << deltaTime << std::endl;
	}
	return foodPos;
}
sf::Vector2i* colonyLocation(Map& map) {
	//std::cout << map.seed << std::endl;
	srand(map.seed);
	int randa = rand() % 4;

	std::cout << "Colony Quadrant: " << randa << std::endl;

	std::chrono::steady_clock::time_point t1;
	std::chrono::steady_clock::time_point t2;
	float deltaTime;
	if(enableTiming)
		t1 = std::chrono::high_resolution_clock::now();
	std::cout << "-----COLONY-----" << std::endl;
	sf::Vector2i* colonyPos =  getValidArea(map, randa);

	if (enableTiming) {
		t2 = std::chrono::high_resolution_clock::now();
		deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(t2 - t1).count();
		std::cout << "   Valid Colony Pos Time: " << deltaTime << std::endl;
	}

	return colonyPos;
	
}

sf::Vector2i* getValidArea(Map& map, int quadrant) {
	int search = std::min((int)(map.height / 2), (int)(map.width / 2));
	if (quadrant == 0) {

		//TL

		/*Search - All follow similar
		  0 1 2 3 4
		0 X X X X
		1 X X X
		2 X X
		3 X
		4
		*/

		for (int dy = 0; dy < search; dy++) {
			int dy2 = dy;
			for (int dx = 0; dx <= dy; dx++) {
				std::cout << dx << "," << dy2 << std::endl;
				if (isValidForColonyAndFood(map, dx, dy2)) {
					//std::cout << "Valid Loc: " << dx << "," << dy2 << std::endl;
					return new sf::Vector2i(dx, dy2);
				}
				dy2--;
			}
		}
	}
	else if (quadrant == 1) {
		//TR
		for (int dy = 0; dy < search; dy++) {
			int dy2 = dy;
			for (int dx = map.width - 1; dx >= map.width - dy; dx--) {
				std::cout << dx << "," << dy2 << std::endl;
				if (isValidForColonyAndFood(map, dx, dy2)) {
					//std::cout << "Valid Loc: " << dx << "," << dy2 << std::endl;
					return new sf::Vector2i(dx, dy2);
				}
				dy2--;
			}
		}
	}
	else if (quadrant == 2) {
		//BR
		int s = map.height - 1;
		for (int dy = map.height - 1; dy >= search; dy--) {
			int dy2 = dy + 1;
			for (int dx = map.width - 1; dx >= map.width - (s - dy); dx--) {
				std::cout << dx << "," << dy2 << std::endl;
				if (isValidForColonyAndFood(map, dx, dy2)) {
					//std::cout << "Valid Loc: " << dx << "," << dy2 << std::endl;
					return new sf::Vector2i(dx, dy2);
				}
				dy2++;
			}
		}
	}
	else {
		//BL
		int s = map.height - 1;
		for (int dy = map.height - 1; dy >= search; dy--) {
			int dy2 = dy;
			for (int dx = 0; dx <= s - dy; dx++) {
				std::cout << dx << "," << dy2 << std::endl;
				if (isValidForColonyAndFood(map, dx, dy2)) {
					//std::cout << "Valid Loc: " << dx << "," << dy2 << std::endl;
					return new sf::Vector2i(dx, dy2);
				}
				dy2++;
			}
		}
	}
}

bool isValidForColonyAndFood(Map& map, int x, int y) {
	if (getMapValueAt(map, x, y) == 0) {
		if (getNeighbourWallCount(map, x, y, 2) == 0) {
			return true;
		}
	}
	return false;
}