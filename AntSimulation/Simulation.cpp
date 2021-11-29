#include "Simulation.hpp";
#include <SFML/Window/Mouse.hpp>
#include "Config.hpp"

Simulation::Simulation() {}


//Main Setups
bool Simulation::loadFromFile(std::string path, bool antiAliasing) {
	sf::Image imgMap;
	if (!imgMap.loadFromFile(path)) {
		return false;
	}
	else {
		std::cout << "Succesfully loaded map: " << path << std::endl;
	}

	

	//Initialisation

	float scaleMultiplier =(imgMap.getSize().x<=80)? (float) 160/ (float) imgMap.getSize().x:2;

	std::cout << "Scale: " << scaleMultiplier << std::endl;

	createColonies();
	
	itemGrid = initItemGrid((int)imgMap.getSize().x*scaleMultiplier, (int) imgMap.getSize().y * scaleMultiplier);
	/*
	for (int i = 0; i < imgMap.getSize().x * scaleMultiplier; i++) {
		for (int j = 0; j < imgMap.getSize().y * scaleMultiplier; j++) {
			sf::Color color;
			color = imgMap.getPixel((int) (i / scaleMultiplier), (int) (j / scaleMultiplier));
			if (color != sf::Color::Black && color != sf::Color::White) {
				int cellIndex = getCellIndex(*itemGrid, (int) i, (int) j);
				Cell& cell = itemGrid->worldCells[cellIndex];
				cell.foodCount = (color.g/25)*5;
			}
		}
	}*/
	
	int mapSizeX = imgMap.getSize().x;
	int mapSizeY = imgMap.getSize().y;

	float worldScaleFromMap = (float)Config::WORLD_SIZE_X / mapSizeX;
	int prevColonyX = -1;
	int prevColonyY = -1;
	int hiveCount = 0;

	cout << worldScaleFromMap << endl;

	if (antiAliasing) {
		for (int i = 0; i < mapSizeX * scaleMultiplier; i++) {
			for (int j = 0; j < mapSizeY * scaleMultiplier; j++) {
				int cX = (int)(i / scaleMultiplier);
				int cY = (int)(j / scaleMultiplier);
				int rX = cX + 1;
				int rY = cY;
				int lX = cX - 1;
				int lY = cY;
				int dX = cX;
				int dY = cY + 1;
				int uX = cX;
				int uY = cY - 1;

				rX = (rX > mapSizeX - 1) ? cX : rX;
				dY = (dY > mapSizeY - 1) ? cY : dY;
				uY = (uY < 0) ? cY : uY;
				lX = (lX < 0) ? cX : lX;


				sf::Color cColor, rColor, dColor, lColor, uColor;
				cColor = imgMap.getPixel(cX, cY);
				rColor = imgMap.getPixel(rX, rY);
				dColor = imgMap.getPixel(dX, dY);
				lColor = imgMap.getPixel(lX, lY);
				uColor = imgMap.getPixel(uX, uY);

				//food
				if (cColor != sf::Color::Black && cColor != sf::Color::White && cColor.g>0) {
					int count = 1;
					int sum = cColor.g;
					int iN = round(i / scaleMultiplier);
					int jN = round(j / scaleMultiplier);
					if (iN > cX && jN > cY) {
						//BR
						if (rColor != sf::Color::Black) {
							count++;
							sum += (rColor == sf::Color::White) ? 0 : rColor.g;

						}
						if (dColor != sf::Color::Black) {
							count++;
							sum += (dColor == sf::Color::White) ? 0 : dColor.g;
						}
					}
					else if (iN > cX) {
						//TR
						if (rColor != sf::Color::Black) {
							count++;
							sum += (rColor == sf::Color::White) ? 0 : rColor.g;
						}
						if (uColor != sf::Color::Black) {
							count++;
							sum += (uColor == sf::Color::White) ? 0 : uColor.g;
						}
					}
					else if (jN > cY) {
						//BL
						if (dColor != sf::Color::Black) {
							count++;
							sum += (dColor == sf::Color::White) ? 0 : dColor.g;
						}
						if (lColor != sf::Color::Black) {
							count++;
							sum += (lColor == sf::Color::White) ? 0 : lColor.g;
						}
					}
					else {
						//TL
						if (uColor != sf::Color::Black) {
							count++;
							sum += (uColor == sf::Color::White) ? 0 : uColor.g;
						}
						if (lColor != sf::Color::Black) {
							count++;
							sum += (lColor == sf::Color::White) ? 0 : lColor.g;
						}
					}
					int greenAvg = sum / count;
					int cellIndex = getCellIndex(*itemGrid, (int)i, (int)j);
					Cell& cell = itemGrid->worldCells[cellIndex];
					//cout << "count: " << count << "| sum: " << sum << "| avg: " << greenAvg << endl;
					cell.foodCount = (greenAvg / 25) * 5;
				}

				//hive
				if (cColor == sf::Color::Blue) {
					
					if (!(prevColonyX == cX && prevColonyY == cY)) {
						cout << "Colony Found: " << cX << "," << cY << endl;
						prevColonyX = cX;
						prevColonyY = cY;

						//cout << (int)(worldScaleFromMap * prevColonyX) << "," << (int)(worldScaleFromMap * prevColonyY);

						updateColony(hiveCount, (int)(worldScaleFromMap * prevColonyX), (int)(worldScaleFromMap * prevColonyY));
					}
				}

				//cout << "c: " << cX << "," << cY << "| r: " << rX << "," << rY << endl;
			}
		}
	}
	else {
		for (int i = 0; i < imgMap.getSize().x * scaleMultiplier; i++) {
			for (int j = 0; j < imgMap.getSize().y * scaleMultiplier; j++) {
				sf::Color color;
				color = imgMap.getPixel((int)(i / scaleMultiplier), (int)(j / scaleMultiplier));
				if (color != sf::Color::Black && color != sf::Color::White) {
					int cellIndex = getCellIndex(*itemGrid, (int)i, (int)j);
					Cell& cell = itemGrid->worldCells[cellIndex];
					cell.foodCount = (color.g / 25) * 5;
				}
			}
		}
	}
	entities = initEntities(colonies, Config::ANT_COUNT);
	map = makeMapPointer(path);
	
	genericSetup();

	return true;
}
void Simulation::generateRandom(bool generateFood) {
	//Initialisation

	createColonies();
	
	itemGrid = initItemGrid(Config::ITEM_GRID_SIZE_X, Config::ITEM_GRID_SIZE_Y);
	map = makeMapPointer(Config::MAP_SIZE_X, Config::MAP_SIZE_Y);
	createMap(map);


	float worldScaleFromMap = (float)Config::WORLD_SIZE_X / map->width;
	float itemGridScaleFromMap = (float)Config::ITEM_GRID_SIZE_X / map->width;

	//Set Food
	sf::Vector2i* foodPos = foodLocation(*map);
	std::cout << "Food pos: " << foodPos->x << "," << foodPos->y << std::endl;
	for (int i = (int)(foodPos->x * itemGridScaleFromMap) - (int)(1*itemGridScaleFromMap); i <= (int)(foodPos->x * itemGridScaleFromMap) + (int)(1 * itemGridScaleFromMap); i++) {
		for (int j = (int)(foodPos->y * itemGridScaleFromMap) - (int)(1 * itemGridScaleFromMap); j <= (int)(foodPos->y * itemGridScaleFromMap) + (int)(1 * itemGridScaleFromMap); j++) {
			int cellIndex = getCellIndex(*itemGrid, i, j);
			Cell& cell = itemGrid->worldCells[cellIndex];
			cell.foodCount = 50;
		}
	}
	
	//Set Colony
	sf::Vector2i* colonyPos = colonyLocation(*map);
	std::cout << "Colony Pos: " << colonyPos->x << ", "  << colonyPos->y << std::endl;

	updateColony(0, (int) (worldScaleFromMap * colonyPos->x), (int) (worldScaleFromMap * colonyPos->y));
	//updateColony(0, 400, 400); 

	

	entities = initEntities(colonies, Config::ANT_COUNT);
	genericSetup();
}

//Loop Methods
void Simulation::updateCellFood(sf::Vector2f mousePos) {
	int cellIndex = getCellIndex(itemGrid, mousePos.x, mousePos.y);
	Cell& cell = itemGrid->worldCells[cellIndex];
	cell.foodCount < 45.0f ? cell.foodCount += 5 : cell.foodCount = 50;
	//food:5 , 0 25 0
	//food:10, 0 51 0
	//food:15, 0 76 0
	//food:20, 0 102 0
	//food:25, 0 127 0
	//food:30, 0 153 0
	//food:35, 0 178 0
	//food:40, 0 204 0
	//food:45, 0 229 0
	//food:50, 0 255 0
}

void Simulation::update(float deltaTime) {
	gridRenderer->update(*itemGrid, deltaTime);
	entityRenderer->update(deltaTime);
	simulateEntitiesOnGPU(entities, itemGrid, map, colonies, deltaTime);
}

void Simulation::render(sf::RenderWindow* window) {
	gridRenderer->render(window);
	entityRenderer->render(window);
	window->draw(*mapArray);

	if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
		sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));
		if (mousePos.x < Config::WORLD_SIZE_X && mousePos.y < Config::WORLD_SIZE_Y && mousePos.x > 0 && mousePos.y > 0) {
			Cell* cell = getCell(itemGrid, mousePos.x, mousePos.y);
			printf("%f, %f\n", cell->pheromones[0], cell->pheromones[1]);
		}
	}
}

//Other Methods
void Simulation::genericSetup() {
	gridRenderer = new GridRenderer(itemGrid,map);
	entityRenderer = new EntityRenderer(entities);

	mapVertices = generateMapVertices(*map);
	map->walls = getBoundariesFromVec2f(getVec2fFromVertices(*mapVertices), mapVertices->size());
	map->wallCount = mapVertices->size() / 2;

	mapArray = getVArrayFromVertices(*mapVertices);
}

void Simulation::createColonies() {
	// COLONIES //
	colonies = createColoniesArray(Config::COLONY_COUNT);
	colonies[0].antCount = 0;
	colonies[0].nestPositionX = 400;
	colonies[0].nestPositionY = 400;
	colonies[0].nestRadius = 10;
}

void Simulation::updateColony(int id, int posX, int posY) {
	colonies[id].nestPositionX = posX;
	colonies[id].nestPositionY = posY;
}