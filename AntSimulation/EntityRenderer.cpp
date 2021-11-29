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
#include "EntityRenderer.hpp"


void EntityRenderer::init() {

	sf::Color defaultColour = sf::Color::Red;
	for (int i = 0; i < entities->entityCount; i++) {
		int vertex = i * 4;
		//vertexArray[i].position = sf::Vector2f((800.0f / grid->worldX)*x + 0.5f, (800.0f / grid->worldY) * y + 0.5f);
		//vertexArray[i].color = defaultColour;
		float x = entities->moves[i].position.x;
		float y = entities->moves[i].position.y;
		vertexArray[vertex].position = sf::Vector2f(x, y);
		vertexArray[vertex + 1].position = sf::Vector2f(x + antSize, y);
		vertexArray[vertex + 2].position = sf::Vector2f(x + antSize, y + antSize);
		vertexArray[vertex + 3].position = sf::Vector2f(x, y + antSize);
		vertexArray[vertex].color = defaultColour;
		vertexArray[vertex + 1].color = defaultColour;
		vertexArray[vertex + 2].color = defaultColour;
		vertexArray[vertex + 3].color = defaultColour;

	}
}
//Threading:
sf::Vector2i EntityRenderer::getEntitiesSet() {
	mutex.lock();
	int x = currentSet;
	if (entitiesRemaining < entitiesPerSet) {
		return sf::Vector2i{ x, entitiesRemaining };
	}
	currentSet += entitiesPerSet;
	entitiesRemaining -= entitiesPerSet;
	mutex.unlock();
	return sf::Vector2i{ x, entitiesPerSet };;
}

void EntityRenderer::vertextDataSet(void* deltaTime){
	sf::Vector2i temp = getEntitiesSet();
	int start = temp.x;
	int end = start + temp.y;
	float deltaTime = *(float*)deltaTime;

	for (int i = start; i < end; i++) {
		int vertex = i * 4;
		//vertexArray[i].position = sf::Vector2f((800.0f / grid->worldX)*x + 0.5f, (800.0f / grid->worldY) * y + 0.5f);
		//vertexArray[i].color = defaultColour;
		float x = entities->moves[i].position.x;
		float y = entities->moves[i].position.y;
		vertexArray[vertex].position = sf::Vector2f(x, y);
		vertexArray[vertex + 1].position = sf::Vector2f(x + antSize, y);
		vertexArray[vertex + 2].position = sf::Vector2f(x + antSize, y + antSize);
		vertexArray[vertex + 3].position = sf::Vector2f(x, y + antSize);
	}
}

void EntityRenderer::render(sf::RenderWindow* window) {
	//std::cout << vertexArray.getVertexCount() << std::endl;
	window->draw(vertexArray);
}

void EntityRenderer::setVertexData(float deltaTime) {
	for (int i = 0; i < entities->entityCount; i++) {
		int vertex = i * 4;
		//vertexArray[i].position = sf::Vector2f((800.0f / grid->worldX)*x + 0.5f, (800.0f / grid->worldY) * y + 0.5f);
		//vertexArray[i].color = defaultColour;
		float x = entities->moves[i].position.x;
		float y = entities->moves[i].position.y;
		vertexArray[vertex].position = sf::Vector2f(x, y);
		vertexArray[vertex + 1].position = sf::Vector2f(x + antSize, y);
		vertexArray[vertex + 2].position = sf::Vector2f(x + antSize, y + antSize);
		vertexArray[vertex + 3].position = sf::Vector2f(x, y + antSize);
	}
}

void EntityRenderer::update(float deltaTime) {
	setVertexData(deltaTime);
}