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
#include "Config.hpp"
#include "EntityRenderer.hpp"
#include "TextRenderer.h"
#include "Simulation.hpp"


void setVertexDataThreaded(sf::VertexArray* vertices, Entities& entities, int threadCount, int threadIndex) {
	int entitiesPerThread = (entities.entityCount / threadCount);
	//cout << "[Thread #" << this_thread::get_id() << "] - Threaded Task #" << threadIndex << "/" << threadCount << " - Job: " << entitiesPerThread << " translations, from " << entitiesPerThread * threadIndex << " to " << (entitiesPerThread * threadIndex) + entitiesPerThread << endl;
	for (int i = entitiesPerThread * threadIndex; i < (entitiesPerThread * threadIndex) + entitiesPerThread; i++) {
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
		tm.queueJob({ 3, false, [&vertices, &entities, &tm, i] { setVertexDataThreaded(vertices, entities, tm.threadCount, i); } });
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

	//TEXT
	TextRenderer tr;


	//SIMULATION
	Simulation simulation;
	if (!simulation.loadFromFile("Maps\\test_map_food_2.png")) {
		exit(EXIT_FAILURE);
	}

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
	//task simEnts = { 2, false, [&entities, &itemGrid, &map, deltaTime] { simulateEntitiesOnGPU(entities, itemGrid, map, deltaTime); } };
	//task drawFrame = { 1, true, [&vertices, &window] {window.draw(vertices); } };

	//TESTING BOUNDARY COLLISION
	/*sf::VertexArray collisionv(sf::Lines, entities->entityCount*4);
	for (int i = 0; i < entities->entityCount*2; i++) {
		collisionv[i].color = sf::Color::Green;
	}*/

	tr.write("FPS", "FPS: ", 20, sf::Vector2f(0.0f, 0.0f));
	while (window.isOpen()) {
		////////////// FPS & DELTATIME //////////////
		deltaTime = deltaClock.restart().asSeconds();
		int fps = 1 / deltaTime;
		//printf("FPS = %d\n", fps);
		tr.update("FPS", TextRenderer::MODIFY_TYPE::TEXT, "FPS: "+to_string(fps));

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
			if (mousePos.x < Config::WORLD_SIZE_X && mousePos.y < Config::WORLD_SIZE_Y && mousePos.x > 0 && mousePos.y > 0) {
				simulation.updateCellFood(mousePos);
			}
		}

		////////////// UPDATE //////////////

		simulation.update(deltaTime);


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

		simulation.render(&window);


		//gridRenderer.render(&window);
		//window.draw(vertices);
		//entityRenderer.render(&window);
		//window.draw(collisionv);
		//window.draw(*mapArray);
		//window.draw(shape, mapTransform);
		//tmanager.queueJob(drawFrame);
		tr.render(window);


		////////////// UPDATE WINDOW //////////////
		window.setView(view);
		window.display();
	}

	return 0;
}
