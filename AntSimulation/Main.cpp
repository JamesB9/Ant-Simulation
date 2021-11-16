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
#include "EntitySystem.cuh"
#include "ItemGrid.cuh"
#include <SFML/Graphics.hpp>
#include <thread>
#include <iostream>
#include "GridRenderer.hpp"

#include "ThreadPoolManager.h"
#include "Render.h"


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

int main() {
	// Window

	sf::RenderWindow window(sf::VideoMode(950, 950), "Ant Colony Simulation");

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
	Entities entities;
	initEntities(entities);
	sf::VertexArray vertices(sf::Points, entities.entityCount);
	for (int i = 0; i < entities.entityCount; i++) {
		vertices[i].color = sf::Color::Red;
	}
	//itemGrid
	ItemGrid itemGrid;
	initItemGrid(itemGrid, 800, 800);
	//renderers
	//Grid
	GridRenderer gridRenderer(itemGrid);
	//Map
	Map map;
	initMap(map, 20, 20);

	// THREADS
	//int threadCount = 10;
	//std::vector<std::thread> threads;
	ThreadPoolManager tmanager;
	task vertexData = { 3, true, [&vertices, &entities] {setVertexData(vertices,entities); } };
	task simEnts = { 2, true, [&entities, &deltaTime] {simulateEntitiesOnGPU(entities, deltaTime); } };
	task drawFrame = { 1, true, [&vertices, &window] {window.draw(vertices); } };

	while (window.isOpen()) {
		// FPS
		deltaTime = deltaClock.restart().asSeconds();
		int fps = 1 / deltaTime;

		// SCREEN CLEAR
		window.clear(sf::Color(10, 10, 10));

		// Process events
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

		//printf("%f -> ", entities.positions[0].x);

		//simulateEntitiesOnGPU(entities, deltaTime);
		//setVertexData(vertices, entities);
		tmanager.queueJob(simEnts);
		//tmanager.join();
		queueVertexData(tmanager, &vertices, entities);
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
		while (!tmanager.queueEmpty()) {}
		window.draw(vertices);
		//tmanager.queueJob(drawFrame);
		//printf("%f\n", entities.positions[0].x);

		printf("%d\n", fps);

		// Update the window
		window.setView(view);
		window.display();
	}

	return 0;
}
