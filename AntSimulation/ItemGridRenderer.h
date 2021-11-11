#pragma once
#include <SFML/Graphics/VertexArray.hpp>;
#include "ItemGrid.cuh"

class GridRenderer {
private:
	ItemGrid* itemGrid;
	sf::Color food = sf::Color(0, 255, 0, 255);
	sf::Color path = sf::Color(255, 0, 0, 255);
public:
	GridRenderer(ItemGrid& itemGrid) { this->itemGrid = &itemGrid; grid = sf::VertexArray(sf::Quads, itemGrid.totalCells * 4); }
	
	
	int calculateVertices();


	sf::Color getCellColour(Cell* cell);
	sf::VertexArray grid;
};
