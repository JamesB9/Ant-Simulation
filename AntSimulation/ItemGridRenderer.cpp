#pragma once
#include "ItemGridRenderer.h"


int GridRenderer::calculateVertices()
{
	float cellSize = 1.0f;
	for (int i = 0; i < itemGrid->totalCells; i += 1) {
		int x, y;
		y = i / itemGrid->worldX;
		x = i - (y * itemGrid->worldX);
		grid[i * 4].position = sf::Vector2f(x, y);
		grid[i * 4 + 1].position = sf::Vector2f(x + cellSize, y);
		grid[i * 4 + 2].position = sf::Vector2f(x + cellSize, y + cellSize);
		grid[i * 4 + 3].position = sf::Vector2f(x, y + cellSize);

		//std::cout << x << ", " << y << std::endl;
		sf::Color thisColor = getCellColour(getCell(*itemGrid, x, y));
		grid[i * 4].color = thisColor;
		grid[i * 4 + 1].color = thisColor;
		grid[i * 4 + 2].color = thisColor;
		grid[i * 4 + 3].color = thisColor;
	}

	return 0;
}

sf::Color GridRenderer::getCellColour(Cell* cell) {
	if (cell->foodCount > 0) {
		return this->food;
	}
	else {
		if (cell->pheromones[0] > cell->pheromones[1]) {
			//Pheromone type 1
			return this->path;
		}
		else if (cell->pheromones[0] < cell->pheromones[1]) {
			//Pheromone type 2
			return this->food;
		}
		else {
			//Equal
			return sf::Color::Black;
		}
	}
}
