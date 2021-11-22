#include "Simulation.hpp";

Simulation::Simulation() {}


//Main Setups
bool Simulation::loadFromFile(std::string path) {
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

	entities = initEntities(Config::ANT_COUNT);
	itemGrid = initItemGrid((int)imgMap.getSize().x*scaleMultiplier, (int) imgMap.getSize().y * scaleMultiplier);

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
	}	

	map = makeMapPointer(path);
	
	genericSetup();

	return true;
}
void Simulation::generateRandom() {
	//Initialisation
	entities = initEntities(Config::ANT_COUNT);
	itemGrid = initItemGrid(Config::ITEM_GRID_SIZE_X, Config::ITEM_GRID_SIZE_Y);
	map = makeMapPointer(Config::MAP_SIZE_X, Config::MAP_SIZE_Y);
	createMap(map);
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
	simulateEntitiesOnGPU(entities, itemGrid, map, deltaTime);
}

void Simulation::render(sf::RenderWindow* window) {
	gridRenderer->render(window);
	entityRenderer->render(window);
	window->draw(*mapArray);
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