#pragma once

////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include "GridRenderer.hpp"
#include "EntitySystem.cuh"
#include "ThreadPoolManager.h"
#include "MarchingSquares.hpp"
#include "Config.hpp"
#include "EntityRenderer.hpp"
#include "Colony.cuh";
#include "TextRenderer.h"
#include <math.h>
#include "SFML/Graphics/CircleShape.hpp"


class Simulation {
public:
	Simulation();

	bool loadFromFile(std::string path, bool antiAliasing=false);
	void generateRandom();

	void update(float deltaTime);
	void updateCellFood(sf::Vector2f mousePos);
	void updateCellPheromone(sf::Vector2f mousePos, int pheromone);

	void render(sf::RenderWindow* window, TextRenderer* tr);

private:
	Colony* colonies;
	Entities* entities;
	ItemGrid* itemGrid;
	Map* map;

	//ThreadPool
	void threadUpdateSim(float deltaTime);


	GridRenderer* gridRenderer;
	EntityRenderer* entityRenderer;

	std::vector<sf::Vector2f>* mapVertices;
	sf::VertexArray* mapArray;

	sf::VertexArray collisionv;

	void createColonies();
	void genericSetup();
};
