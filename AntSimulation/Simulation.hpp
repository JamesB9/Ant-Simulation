#pragma once

#include "GridRenderer.hpp"
#include "EntitySystem.cuh"
#include "ThreadPoolManager.h"
#include "MarchingSquares.hpp"
#include "Config.hpp"
#include "EntityRenderer.hpp"
#include "Colony.cuh";
#include <math.h>

class Simulation {
public:
	Simulation();

	bool loadFromFile(std::string path, bool antiAliasing=false);
	void generateRandom();

	void update(float deltaTime);
	void updateCellFood(sf::Vector2f mousePos);

	void render(sf::RenderWindow* window);

private:
	Colony* colonies;
	Entities* entities;
	ItemGrid* itemGrid;
	Map* map;


	GridRenderer* gridRenderer;
	EntityRenderer* entityRenderer;

	std::vector<sf::Vector2f>* mapVertices;
	sf::VertexArray* mapArray;

	void createColonies();
	void genericSetup();
};
