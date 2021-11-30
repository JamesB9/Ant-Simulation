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
#include "EntitySystem.cuh"
#include <SFML/Window/Mouse.hpp>


////////////////////////////////////////////////////////////
/// \brief Main class for running and rendering the ant simulation
///
////////////////////////////////////////////////////////////
class Simulation {
public:

	////////////////////////////////////////////////////////////
	/// \brief Default Constructor which creates the CPU thread pool
	///
	////////////////////////////////////////////////////////////
	Simulation();


	////////////////////////////////////////////////////////////
	/// \brief Initializes the map, itemgrid and colonies from an image
	///
	/// \param path File path to an image
	/// \param antiAliasing Whether to gradient the food deposits (low amounts near edge)
	///
	////////////////////////////////////////////////////////////
	bool loadFromFile(std::string path, bool antiAliasing=false);


	////////////////////////////////////////////////////////////
	/// \brief Initializes the simulation with randomly a randomly generated environment
	///
	////////////////////////////////////////////////////////////
	void generateRandom();


	////////////////////////////////////////////////////////////
	/// \brief Is called once per timestep to simulate the entities and update vertex data
	///
	/// \param deltaTime The timestep between last update and now
	///
	////////////////////////////////////////////////////////////
	void update(float deltaTime);


	////////////////////////////////////////////////////////////
	/// \brief Adds food to the ItemGrid at the location provided
	///
	/// \param mousePos A 2D vector for where place food
	///
	////////////////////////////////////////////////////////////
	void updateCellFood(sf::Vector2f mousePos);


	////////////////////////////////////////////////////////////
	/// \brief Manually adds pheromone to an ItemGrid cell
	///
	/// \param mousePos A 2D vector for where place pheromone
	/// \param pheromone Type of pheromone to add
	///
	////////////////////////////////////////////////////////////
	void updateCellPheromone(sf::Vector2f mousePos, int pheromone);


	////////////////////////////////////////////////////////////
	/// \brief Render the simulation to the window
	///
	/// \param window Window to render the simulation frame to 
	/// \param tr TextRenderer to use to update onscreen GUI text
	///
	////////////////////////////////////////////////////////////
	void render(sf::RenderWindow* window, TextRenderer* tr);

private:
	Colony* colonies;
	Entities* entities;
	ItemGrid* itemGrid;
	Map* map;


	GridRenderer* gridRenderer;
	EntityRenderer* entityRenderer;

	std::vector<sf::Vector2f>* mapVertices;
	sf::VertexArray* mapArray;

	sf::VertexArray collisionv;

	////////////////////////////////////////////////////////////
	/// \brief Create and Setup the ant colonies
	///
	////////////////////////////////////////////////////////////
	void createColonies();

	////////////////////////////////////////////////////////////
	/// \brief Initializes the renderers and vertex arrays
	///
	////////////////////////////////////////////////////////////
	void genericSetup();
};
