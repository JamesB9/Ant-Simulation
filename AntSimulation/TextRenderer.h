#pragma once
#include <vector>;
#include <map>;
#include <string>;
#include <filesystem>;
#include <iostream>;
#include "SFML/Graphics/Text.hpp"
#include <SFML/Graphics/RenderWindow.hpp>

using namespace std;
namespace fs = std::filesystem;

class TextRenderer {
public:
	const enum MODIFY_TYPE {
		TEXT, FONT, SIZE, POS
	};

	void write(string identifyer, string text, int size, sf::Vector2f position, string font = "Default");
	void update(string identifyer, TextRenderer::MODIFY_TYPE param, string change);

	void render(sf::RenderWindow& window);

	TextRenderer();
private:

	bool fontsAvailable = false;
	string dir = "./Fonts/";
	vector<string> titles;
	map<string, sf::Text> texts;
	map<string, sf::Font*> fonts;
	void loadFonts();
	sf::Font* getFont(string fontName);
};