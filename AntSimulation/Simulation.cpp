#include "Simulation.hpp";

Simulation::Simulation() {}


//Main Setups
void Simulation::loadFromFile(std::string path) {

	//Initialisation
	entities = initEntities(Config::ANT_COUNT);
	itemGrid = initItemGrid(Config::ITEM_GRID_SIZE_X, Config::ITEM_GRID_SIZE_Y);

	//need to set food pos

	map = makeMapPointer(path);
	
	genericSetup();
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