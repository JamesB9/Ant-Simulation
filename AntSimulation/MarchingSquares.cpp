#pragma once

#include "MarchingSquares.hpp"

uint8_t getCase(int tl, int tr, int bl, int br) {
	return 8 * tl + 4 * tr + 2 * br + bl;
}

void addCase1(std::vector<sf::Vector2f>& vertices, float x, float y, float inc) {
	vertices.push_back(sf::Vector2f(x, y + inc / 2));
	vertices.push_back(sf::Vector2f(x + inc / 2, y + inc));
}

void addCase2(std::vector<sf::Vector2f>& vertices, float x, float y, float inc) {
	vertices.push_back(sf::Vector2f(x + inc / 2, y + inc));
	vertices.push_back(sf::Vector2f(x + inc, y + inc / 2));
}

void addCase3(std::vector<sf::Vector2f>& vertices, float x, float y, float inc) {
	vertices.push_back(sf::Vector2f(x, y + inc / 2));
	vertices.push_back(sf::Vector2f(x + inc, y + inc / 2));
}

void addCase4(std::vector<sf::Vector2f>& vertices, float x, float y, float inc) {
	vertices.push_back(sf::Vector2f(x + inc / 2, y));
	vertices.push_back(sf::Vector2f(x + inc, y + inc / 2));
}

void addCase6(std::vector<sf::Vector2f>& vertices, float x, float y, float inc) {
	vertices.push_back(sf::Vector2f(x + inc / 2, y));
	vertices.push_back(sf::Vector2f(x + inc / 2, y + inc));
}

void addCase8(std::vector<sf::Vector2f>& vertices, float x, float y, float inc) {
	vertices.push_back(sf::Vector2f(x, y + inc / 2));
	vertices.push_back(sf::Vector2f(x + inc / 2, y));
}

sf::VertexArray* generateShape(Map& map) {
	//sf::ConvexShape* shape = new sf::ConvexShape();
	std::vector<sf::Vector2f> vertices;


	for (int y = 0; y < map.height; y++) {
		for (int x = 0; x < map.width; x++) {
			int tl = getMapValueAt(map, x, y); // top left
			int tr = (x < map.width - 1) ? getMapValueAt(map, x + 1, y) : 1; // top right
			int bl = (y < map.height - 1) ? getMapValueAt(map, x, y + 1) : 1; // bottom left
			int br = (x < map.width - 1 && y < map.height - 1) ? getMapValueAt(map, x + 1, y + 1) : 1; // bottom right
			/*
			int tl = map[x][y]; // top left
			int tr = (x < map.width - 1) ? map[x + 1][y] : 1; // top right
			int bl = (y < map.height - 1) ? map[x][y + 1] : 1; // bottom left
			int br = (x < map.width - 1 && y < map.height - 1) ? map[x + 1][y + 1] : 1; // bottom right
			*/
			//printf("%d, %d, %d, %d", tl, tr, bl, br);
			float vx = x + 0.5f;
			float vy = y + 0.5f;
			float inc = 1.0f;

			switch (getCase(tl, tr, bl, br)) {
			case 0:
				break;
			case 1:
				// bl triangle
				addCase1(vertices, x, y, inc);
				break;
			case 2:
				// br triangle
				addCase2(vertices, x, y, inc);
				break;
			case 3:
				// bl + br rect
				addCase3(vertices, x, y, inc);
				break;
			case 4:
				// tr triangle
				addCase4(vertices, x, y, inc);
				break;
			case 5:
				addCase8(vertices, x, y, inc);
				addCase2(vertices, x, y, inc);
				break;
			case 6:
				addCase6(vertices, x, y, inc);
				break;
			case 7:
				addCase8(vertices, x, y, inc);
				break;
			case 8:
				addCase8(vertices, x, y, inc);
				break;
			case 9:
				addCase6(vertices, x, y, inc);
				break;
			case 10:
				addCase1(vertices, x, y, inc);
				addCase4(vertices, x, y, inc);
				break;
			case 11:
				addCase4(vertices, x, y, inc);
				break;
			case 12:
				addCase3(vertices, x, y, inc);
				break;
			case 13:
				addCase2(vertices, x, y, inc);
				break;
			case 14:
				addCase1(vertices, x, y, inc);
				break;
			case 15:
				break;
			}
		}
	}

	sf::VertexArray* vArray = new sf::VertexArray(sf::Lines, vertices.size());
	for (sf::Vector2f v : vertices) {
		vArray->append(sf::Vertex(v, sf::Color::White));
	}

	return vArray;
}



