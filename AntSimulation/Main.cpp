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

////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////
// Main Function
////////////////////////////////////////////////////////////
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
	tr.write("FPS", "FPS: ", 20, sf::Vector2f(0.0f, 0.0f));
	tr.write("CELLPOS", "Position: []", 15, sf::Vector2f(0.0f, 25.0f));
	tr.write("CELLINT", "Intensity: []", 15, sf::Vector2f(0.0f, 50.0f));
	tr.write("COLONYFOODCOUNT", "Food In Colony: 0", 15, sf::Vector2f(0.0f, 75.0f));

	//SIMULATION
	Simulation simulation;
	simulation.generateRandom();
	/*if (!simulation.loadFromFile("Maps\\test_map_food_2.png", true)) {
		exit(EXIT_FAILURE);
	}*/

	/*
	sf::Image* image = simulation.generateMapImage();
	sf::Texture texture;
	texture.loadFromImage(*image);
	sf::Sprite sprite;
	sprite.setScale(10, 10);
	sprite.setTexture(texture, true);*/

	////////////// MAIN SIMULATION LOOP //////////////
	while (window.isOpen()) {

		////////////// FPS & DELTATIME //////////////
		deltaTime = deltaClock.restart().asSeconds();
		int fps = 1 / deltaTime;

		//printf("%d, %f\n", frame++, deltaTime);
		//printf("FPS = %d\n", fps);
		tr.update("FPS", TextRenderer::MODIFY_TYPE::TEXT, "FPS: "+to_string(fps));
		tr.update("COLONYFOODCOUNT", TextRenderer::MODIFY_TYPE::TEXT, "Food In Colony: " + to_string(simulation.getFoodCount(0)));


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

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) { // LEFT MOUSE BUTTON
			sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			if (mousePos.x < Config::WORLD_SIZE_X && mousePos.y < Config::WORLD_SIZE_Y && mousePos.x > 0 && mousePos.y > 0) {
				simulation.updateCellFood(mousePos);
			}
		}
		if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) { // RIGHT MOUSE BUTTON
			sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));
			if (mousePos.x < Config::WORLD_SIZE_X && mousePos.y < Config::WORLD_SIZE_Y && mousePos.x > 0 && mousePos.y > 0) {
				tr.update("CELLPOS", TextRenderer::MODIFY_TYPE::TEXT, "Position: [" + to_string(mousePos.x) + ", " + to_string(mousePos.y) + "]");
			}
		}


		////////////// UPDATE //////////////
		simulation.update(deltaTime);


		////////////// RENDERING //////////////
		simulation.render(&window, &tr);
		tr.render(window);
		//window.draw(sprite);


		////////////// UPDATE WINDOW //////////////
		window.setView(view);
		window.display();
	}
	return 0;
}
