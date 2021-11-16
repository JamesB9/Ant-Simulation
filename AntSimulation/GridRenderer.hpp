///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: GridRenderer.hpp
// Description: The headder for the rendering system for the grid
// 
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:  
//	- (URL)
// 
// 
//////////////////////////// 80 columns wide //////////////////////////////////
#include "ItemGrid.cuh"
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Vector3.hpp>
#include <iostream>

class GridRenderer {
public:

	GridRenderer(ItemGrid* grid) : grid{ grid } {
		vertexArray = sf::VertexArray(sf::Quads, grid->totalCells * 4);
		init();
	}

	void render(sf::RenderWindow* window);
	sf::VertexArray& getVertexArray() { return vertexArray; }
	void update(ItemGrid& grid);
private:
	sf::VertexArray vertexArray;
	ItemGrid* grid;

	sf::Vector3f PHEROMONE_0_COLOUR = {0,0,255};
	sf::Vector3f PHEROMONE_1_COLOUR = {255,0,0};

	void init();
};