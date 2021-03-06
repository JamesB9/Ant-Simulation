////////////////////////////////////////////////////////////
/// Headers
////////////////////////////////////////////////////////////
#pragma once
#include "Map.cuh"
#include <SFML/Graphics/ConvexShape.hpp>
#include <vector>
#include "Config.hpp"


////////////////////////////////////////////////////////////
/// \brief Uses Marching Squares algorithm to create map lines
/// 
/// \param map The map to use for generating lines
/// 
/// \return list of 2D vectors, each pair defining a line (map wall)
///
////////////////////////////////////////////////////////////
std::vector<sf::Vector2f>* generateMapVertices(Map& map);

sf::VertexArray* getVArrayFromVertices(std::vector<sf::Vector2f> vertices);

Vec2f* getVec2fFromVertices(std::vector<sf::Vector2f> vertices);

Boundary* getBoundariesFromVec2f(Vec2f* vecs, int totalVecs) {
	Boundary* walls;
	cudaMallocManaged(&walls, (totalVecs/2) * sizeof(Boundary));
	int vecIndex = 0;
	for (int i = 0; i < (totalVecs/2); i++) {
		walls[i] = { vecs[vecIndex++], vecs[vecIndex++], i };
	}

	return walls;
}

