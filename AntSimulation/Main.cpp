///////////////////////////////////////////////////////////////////////////////
// Title:            Ant Simulation
// Authors:           James Sergeant (100301636), James Burling (100266919),
//					  CallumGrimble (100243142) and Oliver Boys (100277126)
// File: Main.cpp
// Description: This is the main driver file for the program.
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
#include <SFML/Graphics.hpp>
#include <thread>
#include <iostream>

#include "GridRenderer.hpp"
#include "EntitySystem.cuh"
#include "ThreadPoolManager.h"
#include "MarchingSquares.hpp"
//#include "MapGenerator.hpp"


void setVertexDataThreaded(sf::VertexArray* vertices, Entities& entities, int threadCount, int threadIndex) {
	int entitiesPerThread = entities.entityCount / threadCount;
	//cout << "Threaded Task #" << threadIndex << "/" << threadCount << " - Job: " << entitiesPerThread << " translations, from " << entitiesPerThread * threadIndex << " to " << (entitiesPerThread * threadIndex) + entitiesPerThread-1 << endl;
	for (int i = entitiesPerThread * threadIndex; i < (entitiesPerThread * threadIndex) + entitiesPerThread-1; i++) {
		(*vertices)[i].position.x = entities.moves[i].position.x;
		(*vertices)[i].position.y = entities.moves[i].position.y;
	}
}

void setVertexData(sf::VertexArray& vertices, Entities& entities) {
	for (int i = 0; i < entities.entityCount; i++) {
		vertices[i].position.x =
			entities.moves[i].position.x;
		vertices[i].position.y =
			entities.moves[i].position.y;
	}
}

void queueVertexData(ThreadPoolManager& tm, sf::VertexArray* vertices, Entities& entities) {
	for (int i = 0; i < tm.threadCount; i++) { // For every thread
		tm.queueJob({ 3, true, [&vertices, &entities, &tm, i] { setVertexDataThreaded(vertices, entities, tm.threadCount, i); } });
	}
}

//Dev testing
void setVertexDataCollision(sf::VertexArray& vertices, Entities& entities) {
	int vertexCounter = 0;
	for (int i = 0; i < entities.entityCount; i++) {
		vertices[vertexCounter].position.x =
			entities.collisions[i].targetPosition.x;
		vertices[vertexCounter].position.y =
			entities.collisions[i].targetPosition.y;
		vertices[vertexCounter +1].position.x =
			entities.moves[i].position.x;
		vertices[vertexCounter +1].position.y =
			entities.moves[i].position.y;

		vertices[vertexCounter+2].position.x =
			entities.collisions[i].targetPosition.x;
		vertices[vertexCounter+2].position.y =
			entities.collisions[i].targetPosition.y;

		vertices[vertexCounter + 3].position.x =
			entities.collisions[i].refractionPosition.x;
		vertices[vertexCounter + 3].position.y =
			entities.collisions[i].refractionPosition.y;
		vertexCounter += 4;
	}
}


int main() {
	// Window

	sf::RenderWindow window(sf::VideoMode(800, 800), "Ant Colony Simulation");

	// Camera
	sf::View view;
	view.setViewport(sf::FloatRect(0.0f, 0.0f, 1.0f, 1.0f));
	view.setSize(sf::Vector2f(800.0f, 800.0f));
	view.setCenter(sf::Vector2f(800.0f / 2.0f, 800.0f / 2.0f));
	window.setView(view);

	// FPS
	sf::Clock deltaClock;
	float deltaTime;


	// SETUP SIMULATION
	//ents
	Entities* entities = initEntities(1000);
	sf::VertexArray vertices(sf::Points, entities->entityCount);
	for (int i = 0; i < entities->entityCount; i++) {
		vertices[i].color = sf::Color::Red;
	}
	//itemGrid
	ItemGrid* itemGrid = initItemGrid(160, 160);

	//Map
	Map* map = makeMapPointer(80, 80);
	createMap(map);

	//renderers
	//Grid
	GridRenderer gridRenderer(itemGrid, map);

	//for (int x = 0; x < map.width; x++) {
	//	for (int y = 0; y < map.height; y++) {
	//		std::cout << getMapValueAt(map, x, y) << " = " << map[x][y] << std::endl;
	//	}
	//}
	std::vector<sf::Vector2f>* mapVertices = generateMapVertices(*map);

	/*mapVertices->push_back({0.0f, 0.0f}); //tl
	mapVertices->push_back({ 0.0f, 80.0f });//bl
	mapVertices->push_back({ 0.0f, 80.0f });//bl
	mapVertices->push_back({ 80.0f, 80.0f });//br
	mapVertices->push_back({ 80.0f, 80.0f });//br
	mapVertices->push_back({ 80.0f, 0.0f });//tr
	mapVertices->push_back({ 80.0f, 0.0f });//tr
	mapVertices->push_back({ 0.0f, 0.0f });//tl*/

	map->walls = getBoundariesFromVec2f(getVec2fFromVertices(*mapVertices), mapVertices->size());
	map->wallCount = mapVertices->size() / 2;

	sf::VertexArray* mapArray = getVArrayFromVertices(*mapVertices);
	sf::Transform mapTransform;
	//mapTransform.scale(10, 10);
	/*
	sf::ConvexShape shape = sf::ConvexShape(mapArray->getVertexCount());
	for (int i = 0; i < mapArray->getVertexCount(); i++) {
		shape.setPoint(i, (*mapArray)[i].position);
	}*/

	// THREADS
	//int threadCount = 10;
	//std::vector<std::thread> threads;
	//ThreadPoolManager tmanager;
	//task vertexData = { 3, true, [&vertices, &entities] {setVertexData(vertices,entities); } };
	//task simEnts = { 2, true, [&entities, &itemGrid, &deltaTime] {simulateEntitiesOnGPU(entities, itemGrid, deltaTime); } };
	//task drawFrame = { 1, true, [&vertices, &window] {window.draw(vertices); } };

	//TESTING BOUNDARY COLLISION
	sf::VertexArray collisionv(sf::Lines, entities->entityCount*4);
	for (int i = 0; i < entities->entityCount*2; i++) {
		collisionv[i].color = sf::Color::Green;
	}

	while (window.isOpen()) {
		////////////// FPS & DELTATIME //////////////
		deltaTime = deltaClock.restart().asSeconds();
		int fps = 1 / deltaTime;
		printf("FPS = %d\n", fps);

		////////////// CLEAR SCREEN //////////////
		window.clear(sf::Color(10, 10, 10));

		////////////// PROCESS EVENTS //////////////
		sf::Event event;
		while (window.pollEvent(event))
		{
			// Close window: exit
			if (event.type == sf::Event::Closed) {
				window.close();
			}
			// catch the resize events
			if (event.type == sf::Event::Resized)
			{
				// update the view to the new size of the window
				sf::FloatRect visibleArea(0.f, 0.f, event.size.width, event.size.height);
				view.setSize(sf::Vector2f(event.size.width, event.size.height));
			}

		}

		////////////// CONTROLS //////////////
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) deltaTime = 0; // Pause Simulation
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) view.move(sf::Vector2f(-deltaTime * 100.0f, 0.0f));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) view.move(sf::Vector2f(deltaTime * 100.0f, 0.0f));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) view.move(sf::Vector2f(0.0f, -deltaTime * 100.0f));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) view.move(sf::Vector2f(0.0f, deltaTime * 100.0f));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) view.zoom(1 + (deltaTime * -2.0f));
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::O)) view.zoom(1 + (deltaTime * 2.0f));
		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
			sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			Cell* cell = getCell(itemGrid, mousePos.x, mousePos.y);
			cell->foodCount += 1.0f;
		}

		////////////// PHEROMONE & FOOD RENDERING //////////////
		gridRenderer.update(*itemGrid, deltaTime);


		////////////// SIMULATION //////////////
		simulateEntitiesOnGPU(entities, itemGrid, map, deltaTime);
		setVertexData(vertices, *entities);
		//setVertexDataCollision(collisionv, entities);





		//simulateEntitiesOnGPU(entities, deltaTime);
		//setVertexData(vertices, entities);
		//tmanager.queueJob(simEnts);
		//tmanager.join();
		//queueVertexData(tmanager, &vertices, entities);
		//tmanager.queueJob(vertexData);
		/*
		for (int i = 0; i < threadCount; i++) {
			threads.push_back(std::thread(setVertexDataThreaded, &vertices, &entities, threadCount, i));
		}

		for (auto& th : threads) {
			th.join();
		}
		threads.clear();
		*/
		//printf("%f, %f\n", vertices[0].position.x, vertices[0].position.y);
		//while (!tmanager.queueEmpty()) {}


		////////////// DRAWING //////////////
		gridRenderer.render(&window);
		window.draw(vertices);
		//window.draw(collisionv);
		window.draw(*mapArray, mapTransform);
		//window.draw(shape, mapTransform);
		//tmanager.queueJob(drawFrame);

		////////////// UPDATE WINDOW //////////////
		window.setView(view);
		window.display();
	}

	return 0;
}
