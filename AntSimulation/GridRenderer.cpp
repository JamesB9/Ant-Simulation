///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919), 
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: GridRenderer.cpp
// Description: The rendering system for the grid
// 
// Change Log:
//	- 15/11/2021:JS - Added in block comments.
//
// Online sources:  
//	- (URL)
// 
// 
//////////////////////////// 80 columns wide //////////////////////////////////
#include "GridRenderer.hpp"
#include <SFML/Graphics/RenderWindow.hpp>

void GridRenderer::init() {
	float cellSize = 1.0f;
	sf::Color defaultColour = { 10, 10, 10, 255 };
	for (int i = 0; i < grid.totalCells; i++) {
		int y = i / grid.worldX;
		int x = i - (grid.worldX * y);
		int vertex = i * 4;

		vertexArray[vertex].position = sf::Vector2f(x, y );
		vertexArray[vertex + 1].position = sf::Vector2f(x + cellSize, y );
		vertexArray[vertex + 2].position = sf::Vector2f(x + cellSize, y + cellSize );
		vertexArray[vertex + 3].position = sf::Vector2f(x, y + cellSize );
		vertexArray[vertex].color = defaultColour;
		vertexArray[vertex + 1].color = defaultColour;
		vertexArray[vertex + 2].color = defaultColour;
		vertexArray[vertex + 3].color = defaultColour;
	}
}

void GridRenderer::render(sf::RenderWindow* window) {
	window->draw(vertexArray);
}