#include <iostream>
#include <SFML/Graphics.hpp>

#include "Entities.hpp"

int main() {
	// Window
	sf::RenderWindow window(sf::VideoMode(1600, 1600), "Ant Colony Simulation");

	// Camera
	sf::View view;
	view.setViewport(sf::FloatRect(0.0f, 0.0f, 1.0f, 1.0f));
	view.setSize(sf::Vector2f(800.0f, 800.0f));
	view.setCenter(sf::Vector2f(800.0f / 2.0f, 800.0f / 2.0f));
	window.setView(view);


	// SETUP SIMULATION
	Entities entities;

	EntityID antID = entities.addEntity();
	entities.positions[antID].x = 10;
	entities.positions[antID].y = 10;

	entities.moves[antID].velx = 5;

	
	while (true) {
		window.display();
	}
	
	return 0;
}