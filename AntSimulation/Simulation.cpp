#include "Simulation.hpp";
#include <SFML/Window/Mouse.hpp>

Simulation::Simulation() {}


//Main Setups
bool Simulation::loadFromFile(std::string path, bool antiAliasing) {
	sf::Image imgMap;
	if (!imgMap.loadFromFile(path)) {
		return false;
	}
	else {
		std::cout << "Succesfully loaded map: " << path << std::endl;
	}


	//Initialisation

	float scaleMultiplier =(imgMap.getSize().x<=80)? (float) 160/ (float) imgMap.getSize().x:2;

	std::cout << "Scale: " << scaleMultiplier << std::endl;

	createColonies();
	entities = initEntities(colonies, Config::ANT_COUNT);
	itemGrid = initItemGrid((int)imgMap.getSize().x*scaleMultiplier, (int) imgMap.getSize().y * scaleMultiplier);
	/*
	for (int i = 0; i < imgMap.getSize().x * scaleMultiplier; i++) {
		for (int j = 0; j < imgMap.getSize().y * scaleMultiplier; j++) {
			sf::Color color;
			color = imgMap.getPixel((int) (i / scaleMultiplier), (int) (j / scaleMultiplier));
			if (color != sf::Color::Black && color != sf::Color::White) {
				int cellIndex = getCellIndex(*itemGrid, (int) i, (int) j);
				Cell& cell = itemGrid->worldCells[cellIndex];
				cell.foodCount = (color.g/25)*5;
			}
		}
	}*/
	
	int mapSizeX = imgMap.getSize().x;
	int mapSizeY = imgMap.getSize().y;

	if (antiAliasing) {

		for (int i = 0; i < mapSizeX * scaleMultiplier; i++) {
			for (int j = 0; j < mapSizeY * scaleMultiplier; j++) {
				int cX = (int)(i / scaleMultiplier);
				int cY = (int)(j / scaleMultiplier);
				int rX = cX + 1;
				int rY = cY;
				int lX = cX - 1;
				int lY = cY;
				int dX = cX;
				int dY = cY + 1;
				int uX = cX;
				int uY = cY - 1;

				rX = (rX > mapSizeX - 1) ? cX : rX;
				dY = (dY > mapSizeY - 1) ? cY : dY;
				uY = (uY < 0) ? cY : uY;
				lX = (lX < 0) ? cX : lX;


				sf::Color cColor, rColor, dColor, lColor, uColor;
				cColor = imgMap.getPixel(cX, cY);
				rColor = imgMap.getPixel(rX, rY);
				dColor = imgMap.getPixel(dX, dY);
				lColor = imgMap.getPixel(lX, lY);
				uColor = imgMap.getPixel(uX, uY);


				if (cColor != sf::Color::Black && cColor != sf::Color::White) {
					int count = 1;
					int sum = cColor.g;
					int iN = round(i / scaleMultiplier);
					int jN = round(j / scaleMultiplier);
					if (iN > cX && jN > cY) {
						//BR
						if (rColor != sf::Color::Black) {
							count++;
							sum += (rColor == sf::Color::White) ? 0 : rColor.g;

						}
						if (dColor != sf::Color::Black) {
							count++;
							sum += (dColor == sf::Color::White) ? 0 : dColor.g;
						}
					}
					else if (iN > cX) {
						//TR
						if (rColor != sf::Color::Black) {
							count++;
							sum += (rColor == sf::Color::White) ? 0 : rColor.g;
						}
						if (uColor != sf::Color::Black) {
							count++;
							sum += (uColor == sf::Color::White) ? 0 : uColor.g;
						}
					}
					else if (jN > cY) {
						//BL
						if (dColor != sf::Color::Black) {
							count++;
							sum += (dColor == sf::Color::White) ? 0 : dColor.g;
						}
						if (lColor != sf::Color::Black) {
							count++;
							sum += (lColor == sf::Color::White) ? 0 : lColor.g;
						}
					}
					else {
						//TL
						if (uColor != sf::Color::Black) {
							count++;
							sum += (uColor == sf::Color::White) ? 0 : uColor.g;
						}
						if (lColor != sf::Color::Black) {
							count++;
							sum += (lColor == sf::Color::White) ? 0 : lColor.g;
						}
					}
					int greenAvg = sum / count;
					int cellIndex = getCellIndex(*itemGrid, (int)i, (int)j);
					Cell& cell = itemGrid->worldCells[cellIndex];
					//cout << "count: " << count << "| sum: " << sum << "| avg: " << greenAvg << endl;
					cell.foodCount = (greenAvg / 25) * 5;
				}



				//cout << "c: " << cX << "," << cY << "| r: " << rX << "," << rY << endl;
			}
		}
	}
	else {
		for (int i = 0; i < imgMap.getSize().x * scaleMultiplier; i++) {
			for (int j = 0; j < imgMap.getSize().y * scaleMultiplier; j++) {
				sf::Color color;
				color = imgMap.getPixel((int)(i / scaleMultiplier), (int)(j / scaleMultiplier));
				if (color != sf::Color::Black && color != sf::Color::White) {
					int cellIndex = getCellIndex(*itemGrid, (int)i, (int)j);
					Cell& cell = itemGrid->worldCells[cellIndex];
					cell.foodCount = (color.g / 25) * 5;
				}
			}
		}
	}

	map = makeMapPointer(path);
	
	genericSetup();

	return true;
}
void Simulation::generateRandom() {
	//Initialisation

	createColonies();
	entities = initEntities(colonies, Config::ANT_COUNT);
	itemGrid = initItemGrid(Config::ITEM_GRID_SIZE_X, Config::ITEM_GRID_SIZE_Y);
	map = makeMapPointer(Config::MAP_SIZE_X, Config::MAP_SIZE_Y);
	createMap(map);
	genericSetup();
}

//Loop Methods
void Simulation::updateCellFood(sf::Vector2f mousePos) {
	int cellIndex = getCellIndex(itemGrid, mousePos.x, mousePos.y);
	Cell& cell = itemGrid->worldCells[cellIndex];
	//cell.foodCount < 45.0f ? cell.foodCount += 5 : cell.foodCount = 50;
	cell.foodCount = 50;
	//food:5 , 0 25 0
	//food:10, 0 51 0
	//food:15, 0 76 0
	//food:20, 0 102 0
	//food:25, 0 127 0
	//food:30, 0 153 0
	//food:35, 0 178 0
	//food:40, 0 204 0
	//food:45, 0 229 0
	//food:50, 0 255 0
}

void Simulation::updateCellPheromone(sf::Vector2f mousePos, int pheromone) {
	int cellIndex = getCellIndex(itemGrid, mousePos.x, mousePos.y);
	Cell& cell = itemGrid->worldCells[cellIndex];
	cell.pheromones[pheromone] += 1.0f;
	//food:5 , 0 25 0
	//food:10, 0 51 0
	//food:15, 0 76 0
	//food:20, 0 102 0
	//food:25, 0 127 0
	//food:30, 0 153 0
	//food:35, 0 178 0
	//food:40, 0 204 0
	//food:45, 0 229 0
	//food:50, 0 255 0
}

void Simulation::update(float deltaTime) {
	gridRenderer->update(*itemGrid, deltaTime);
	entityRenderer->update(deltaTime);
	simulateEntitiesOnGPU(entities, itemGrid, map, colonies, deltaTime);
}

//Dev testing
void setVertexDataCollision(sf::VertexArray& vertices, Entities& entities) {
	int vertexCounter = 0;
	for (int i = 0; i < entities.entityCount; i++) {
		vertices[vertexCounter].position.x =
			entities.collisions[i].targetPosition.x;
		vertices[vertexCounter].position.y =
			entities.collisions[i].targetPosition.y;
		vertices[vertexCounter + 1].position.x =
			entities.moves[i].position.x;
		vertices[vertexCounter + 1].position.y =
			entities.moves[i].position.y;

		vertices[vertexCounter + 2].position.x =
			entities.collisions[i].targetPosition.x;
		vertices[vertexCounter + 2].position.y =
			entities.collisions[i].targetPosition.y;

		vertices[vertexCounter + 3].position.x =
			entities.collisions[i].refractionPosition.x;
		vertices[vertexCounter + 3].position.y =
			entities.collisions[i].refractionPosition.y;
		vertexCounter += 4;
	}
}

void Simulation::render(sf::RenderWindow* window, TextRenderer* tr) {

	//setVertexDataCollision(this->collisionv, *entities);
	//window->draw(collisionv);

	gridRenderer->render(window);
	entityRenderer->render(window);
	window->draw(*mapArray);

	if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
		sf::Vector2f mousePos = window->mapPixelToCoords(sf::Mouse::getPosition(*window));
		if (mousePos.x < Config::WORLD_SIZE_X && mousePos.y < Config::WORLD_SIZE_Y && mousePos.x > 0 && mousePos.y > 0) {
			Cell* cell = getCell(itemGrid, mousePos.x, mousePos.y);
			//printf("%f, %f\n", cell->pheromones[0], cell->pheromones[1]);
			tr->update("CELLPOS", TextRenderer::MODIFY_TYPE::TEXT, "Something");
			tr->update("CELLINT", TextRenderer::MODIFY_TYPE::TEXT, "Intensity: [" + to_string(cell->pheromones[0]) +","+ to_string(cell->pheromones[1]) + "] \nFood Count: " + to_string(cell->foodCount));
		}
	}

	sf::CircleShape circle = sf::CircleShape(5);

	circle.setFillColor(sf::Color::Magenta);
	for (int i = 0; i < Config::COLONY_COUNT; i++) {
		circle.setRadius(colonies[i].nestRadius);
		circle.setOrigin({ colonies[i].nestRadius/2.0f, colonies[i].nestRadius / 2.0f });
		circle.setPosition({ colonies[i].nestPositionX, colonies[i].nestPositionY });
		window->draw(circle);
	}
}

//Other Methods
void Simulation::genericSetup() {
	gridRenderer = new GridRenderer(itemGrid,map);
	entityRenderer = new EntityRenderer(entities);

	mapVertices = generateMapVertices(*map);
	map->walls = getBoundariesFromVec2f(getVec2fFromVertices(*mapVertices), mapVertices->size());
	map->wallCount = mapVertices->size() / 2;

	mapArray = getVArrayFromVertices(*mapVertices);

	//TESTING BOUNDARY COLLISION
	collisionv = sf::VertexArray(sf::Lines, entities->entityCount * 4);
	for (int i = 0; i < entities->entityCount * 2; i++) {
		collisionv[i].color = sf::Color::Green;
	}
}

void Simulation::createColonies() {
	// COLONIES //
	colonies = createColoniesArray(Config::COLONY_COUNT);
	colonies[0].antCount = 0;
	colonies[0].nestPositionX = 400;
	colonies[0].nestPositionY = 400;
	colonies[0].nestRadius = 10;
}