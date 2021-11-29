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


void GridRenderer::init() {
	float cellSizeX = Config::WORLD_SIZE_X / (float)grid->sizeX;
	float cellSizeY = Config::WORLD_SIZE_Y / (float)grid->sizeY;

	sf::Color defaultColour = { 10, 10, 10, 255 };
	for (int i = 0; i < grid->totalCells; i++) {
		float y = (i / grid->sizeX);
		float x = (i - (grid->sizeX * y));

		int vertex = i * 4;
		//vertexArray[i].position = sf::Vector2f((800.0f / grid->worldX)*x + 0.5f, (800.0f / grid->worldY) * y + 0.5f);
		//vertexArray[i].color = defaultColour;
		x *= cellSizeX;
		y *= cellSizeY;

		vertexArray[vertex].position = sf::Vector2f(x, y);
		vertexArray[vertex + 1].position = sf::Vector2f(x + cellSizeX, y );
		vertexArray[vertex + 2].position = sf::Vector2f(x + cellSizeX, y + cellSizeY );
		vertexArray[vertex + 3].position = sf::Vector2f(x, y + cellSizeY );
		vertexArray[vertex].color = defaultColour;
		vertexArray[vertex + 1].color = defaultColour;
		vertexArray[vertex + 2].color = defaultColour;
		vertexArray[vertex + 3].color = defaultColour;
		
	}
}

void GridRenderer::render(sf::RenderWindow* window) {
	//std::cout << vertexArray.getVertexCount() << std::endl;
	window->draw(vertexArray);
}

float clip(float n, float lower, float upper) {
	return std::max(lower, std::min(n, upper));
}

void GridRenderer::update(ItemGrid& grid, float deltaTime) {
	sf::Vector3f defaultColour = { 10, 10, 10 };
	sf::Color cellColour;
	//Cell& cell = grid.worldCells[0];
	//Cell* cells = grid->worldCells;
	for (int x = 0; x < grid.sizeX; x++) {
		for (int y = 0; y < grid.sizeY; y++) {

			int cellIndex = getCellIndex(grid, x, y);
			Cell& cell = grid.worldCells[cellIndex];
			updateCell(cell, deltaTime);
			int vertex = cellIndex * 4;

			float intensity;
			if (cell.foodCount > 0.0f) { // HAS FOOD
				cellColour = sf::Color(255, 204, 0, (cell.foodCount / 50.0f)*255);
			}
			else { // HASN'T GOT FOOD1
				intensity = 255 * clip(cell.pheromones[0] + cell.pheromones[1], 0, 0.5);
				sf::Vector3f cellColourV = (cell.pheromones[0] * PHEROMONE_0_COLOUR) + (cell.pheromones[1] * PHEROMONE_1_COLOUR);
				cellColour = sf::Color(clip(cellColourV.x, 0, 255), clip(cellColourV.y, 0, 255), clip(cellColourV.z, 0, 255), intensity);
			}

			vertexArray[vertex].color = cellColour;
			vertexArray[vertex + 1].color = cellColour;
			vertexArray[vertex + 2].color = cellColour;
			vertexArray[vertex + 3].color = cellColour;
		}
	}
}