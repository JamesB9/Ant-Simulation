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
	float cellSize = 1.0f;
	sf::Color defaultColour = { 10, 10, 10, 255 };
	for (int i = 0; i < grid->totalCells; i++) {
		int y = i / grid->worldX;
		int x = i - (grid->worldX * y);
		int vertex = i * 4;

		vertexArray[i].position = sf::Vector2f(x + 0.5f, y + 0.5f);
		vertexArray[i].color = defaultColour;
		/*
		vertexArray[vertex].position = sf::Vector2f(x, y );
		vertexArray[vertex + 1].position = sf::Vector2f(x + cellSize, y );
		vertexArray[vertex + 2].position = sf::Vector2f(x + cellSize, y + cellSize );
		vertexArray[vertex + 3].position = sf::Vector2f(x, y + cellSize );
		vertexArray[vertex].color = defaultColour;
		vertexArray[vertex + 1].color = defaultColour;
		vertexArray[vertex + 2].color = defaultColour;
		vertexArray[vertex + 3].color = defaultColour;
		*/
	}
}

void GridRenderer::render(sf::RenderWindow* window) {
	//std::cout << vertexArray.getVertexCount() << std::endl;
	window->draw(vertexArray);
}

float clip(float n, float lower, float upper) {
	return std::max(lower, std::min(n, upper));
}

void GridRenderer::update(ItemGrid& grid) {
	sf::Color cellColour;
	sf::Vector3f defaultColour = { 10, 10, 10 };
	//Cell& cell = grid.worldCells[0];
	//Cell* cells = grid->worldCells;
	for (int x = 0; x < grid.worldX; x++) {
		for (int y = 0; y < grid.worldY; y++) {

			int cellIndex = getCellIndex(grid, x, y);
			Cell& cell = grid.worldCells[cellIndex];
			//Cell& cell = grid.worldCells[cellIndex];
			sf::Vector3f cellColourV = (cell.pheromones[0] * PHEROMONE_0_COLOUR) + (cell.pheromones[1] * PHEROMONE_1_COLOUR);
			//printf("%f", cell.ph)
			//cellColourV.x = clip(cellColourV.x, defaultColour.x, 255);
			//cellColourV.y = clip(cellColourV.y, defaultColour.y, 255);
			//cellColourV.z = clip(cellColourV.z, defaultColour.z, 255);
			cellColour = sf::Color(cellColourV.x, cellColourV.y, cellColourV.z);
			vertexArray[cellIndex].color = cellColour;
			//int cellIndex = getCellIndex(grid, x, y);
			//int vertex = cellIndex * 4;
			//std::cout << cellIndex << std::endl;
			//Cell& cell = grid.worldCells[cellIndex];
			//float f = grid.worldCells[0].foodCount;
			//std::cout << grid.worldCells[0].foodCount << std::endl;
			//float f = cell.pheromones[0];
			//sf::Vector3f cellColourV = sf::Vector3f(((float)cell.pheromones[0] * PHEROMONE_0_COLOUR));
			/*
			sf::Vector3f cellColourV = (cell.pheromones[0] * PHEROMONE_0_COLOUR) + (cell.pheromones[1] * PHEROMONE_1_COLOUR);
			cellColourV += defaultColour;

			cellColour = sf::Color(clip(cellColourV.x, 0, 255), clip(cellColourV.y, 0, 255), clip(cellColourV.z, 0, 255));
			//cellColour = sf::Color::Green;
			vertexArray[vertex].color = cellColour;
			vertexArray[vertex + 1].color = cellColour;
			vertexArray[vertex + 2].color = cellColour;
			vertexArray[vertex + 3].color = cellColour;*/
		}
	}
}