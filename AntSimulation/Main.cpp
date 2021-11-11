#include "EntitySystem.cuh"
#include <SFML/Graphics.hpp>
#include <thread>

void setVertexDataThreaded(sf::VertexArray* vertices, Entities* entities, int threadCount, int threadIndex) {
	int entitiesPerThread = entities->entityCount / threadCount;
	for (int i = entitiesPerThread * threadIndex; i < (entitiesPerThread * threadIndex) + entitiesPerThread; i++) {
		(*vertices)[i].position.x = entities->moves[i].x;
		(*vertices)[i].position.y = entities->moves[i].y;
	}
}

void setVertexData(sf::VertexArray* vertices, Entities* entities) {
	for (int i = 0; i < entities->entityCount; i++) {
		(*vertices)[i].position.x = entities->moves[i].x;
		(*vertices)[i].position.y = entities->moves[i].y;
	}
}

int main() {
	// Window
	
	sf::RenderWindow window(sf::VideoMode(1600, 1600), "Ant Colony Simulation");

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
	Entities entities;
	initEntities(entities);
	sf::VertexArray vertices(sf::Points, entities.entityCount);
	for (int i = 0; i < entities.entityCount; i++) {
		vertices[i].color = sf::Color::Red;
	}

	// THREADS
	int threadCount = 10;
	std::vector<std::thread> threads;

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
		simulateEntitiesOnGPU(entities, deltaTime);
		setVertexData(&vertices,&entities);
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

		window.draw(vertices);
		//printf("%f\n", entities.positions[0].x);

		printf("%d\n", fps);

		// Update the window
		window.setView(view);
		window.display();
	}

	return 0;
}