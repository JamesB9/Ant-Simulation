#include "ItemGrid.cuh"
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/RenderWindow.hpp>

class GridRenderer {
public:

	GridRenderer(ItemGrid& grid) : grid{ grid } {
		vertexArray = sf::VertexArray(sf::Quads, grid.totalCells * 4);
		init();
	}

	void render(sf::RenderWindow* window);
	sf::VertexArray& getVertexArray() { return vertexArray; }
private:
	sf::VertexArray vertexArray;
	ItemGrid& grid;

	void init();
};