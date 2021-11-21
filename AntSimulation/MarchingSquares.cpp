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

std::vector<sf::Vector2f>* generateMapVertices(Map& map) {
	std::vector<sf::Vector2f>* vertices = new std::vector<sf::Vector2f>();


	for (int y = 0; y < map.height; y++) {
		for (int x = 0; x < map.width; x++) {
			int tl = getMapValueAt(map, x, y); // top left
			int tr = (x < map.width - 1) ? getMapValueAt(map, x + 1, y) : 1; // top right
			int bl = (y < map.height - 1) ? getMapValueAt(map, x, y + 1) : 1; // bottom left
			int br = (x < map.width - 1 && y < map.height - 1) ? getMapValueAt(map, x + 1, y + 1) : 1; // bottom right

			float vx = x * (Config::WORLD_SIZE_X /map.width);
			float vy = y * (Config::WORLD_SIZE_Y / map.height);
			float inc = (Config::WORLD_SIZE_X / map.width);

			switch (getCase(tl, tr, bl, br)) {
			case 0:
				break;
			case 1:
				// bl triangle
				addCase1(*vertices, vx, vy, inc);
				break;
			case 2:
				// br triangle
				addCase2(*vertices, vx, vy, inc);
				break;
			case 3:
				// bl + br rect
				addCase3(*vertices, vx, vy, inc);
				break;
			case 4:
				// tr triangle
				addCase4(*vertices, vx, vy, inc);
				break;
			case 5:
				addCase8(*vertices, vx, vy, inc);
				addCase2(*vertices, vx, vy, inc);
				break;
			case 6:
				addCase6(*vertices, vx, vy, inc);
				break;
			case 7:
				addCase8(*vertices, vx, vy, inc);
				break;
			case 8:
				addCase8(*vertices, vx, vy, inc);
				break;
			case 9:
				addCase6(*vertices, vx, vy, inc);
				break;
			case 10:
				addCase1(*vertices, vx, vy, inc);
				addCase4(*vertices, vx, vy, inc);
				break;
			case 11:
				addCase4(*vertices, vx, vy, inc);
				break;
			case 12:
				addCase3(*vertices, vx, vy, inc);
				break;
			case 13:
				addCase2(*vertices, vx, vy, inc);
				break;
			case 14:
				addCase1(*vertices, vx, vy, inc);
				break;
			case 15:
				break;
			}
		}
	}

	return vertices;
}

sf::VertexArray* getVArrayFromVertices(std::vector<sf::Vector2f> vertices) {
	sf::VertexArray* vArray = new sf::VertexArray(sf::Lines, vertices.size());
	for (sf::Vector2f v : vertices) {
		vArray->append(sf::Vertex(v, sf::Color::White));
	}

	return vArray;
}

Vec2f* getVec2fFromVertices(std::vector<sf::Vector2f> vertices) {
	Vec2f* vArray = (Vec2f*) malloc(vertices.size() * sizeof(Vec2f));
	for (int i = 0; i < vertices.size(); i++) {
		vArray[i] = {vertices[i].x, vertices[i].y};
	}

	return vArray;
}
