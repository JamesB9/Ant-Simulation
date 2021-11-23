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
#include "ThreadPool.hpp"
#include <iostream>
#include <mutex>
#include <math.h>

class EntityRenderer {
public:

	EntityRenderer(Entities* entities, ThreadPool* threadPool) : entities{ entities }, threadPool{threadPool}{
		vertexArray = sf::VertexArray(sf::Quads, entities->entityCount * 4);
		//vertexArray = sf::VertexArray(sf::Points, grid->totalCells);
		init();
	}
	void setVertexData(float deltaTime);
	void render(sf::RenderWindow* window);
	sf::VertexArray& getVertexArray() { return vertexArray; }
	void update(float deltaTime);
	sf::Vector2i getEntitiesSet();
	void vertextDataSet(void*);
private:
	//Threads:
	volatile int currentSet = 0;
	volatile int entitiesRemaining = entities->entityCount;
	int entitiesPerSet = ceil((float)entities->entityCount/threadPool->NUMBER_OF_THREADS);
	std::mutex mutex;
	sf::VertexArray vertexArray;
	Entities* entities;
	float antSize = 1.0f;

	ThreadPool* threadPool;

	void init();
};