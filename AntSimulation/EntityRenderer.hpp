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
#pragma once
#include "Entities.cuh"
#include <SFML/Graphics/VertexArray.hpp>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Vector3.hpp>
#include <iostream>

class EntityRenderer {
public:

	EntityRenderer(Entities* entities) : entities{ entities }{
		vertexArray = sf::VertexArray(sf::Quads, entities->entityCount * 4);
		//vertexArray = sf::VertexArray(sf::Points, grid->totalCells);
		init();
	}

	void render(sf::RenderWindow* window);
	sf::VertexArray& getVertexArray() { return vertexArray; }
	void update(float deltaTime);
private:
	sf::VertexArray vertexArray;
	Entities* entities;
	float antSize = 1.0f;

	void init();
};