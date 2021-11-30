#include "TextRenderer.h";


TextRenderer::TextRenderer() {
	loadFonts();
}

void TextRenderer::write(string identifyer, string text, int size, sf::Vector2f position, string font) {
	//If no fonts are available, cancel writing of text to screen.
	if (!fontsAvailable) { printf("NO FONTS AVAILABLE, stopped text rendering\n"); return; }

	//Push unique identifyer to the list
	titles.push_back(identifyer);
	//Insert a new sf::Text to the texts map
	texts.insert({ identifyer, sf::Text(text, *getFont(font), size) });
	//Find the just now stored sf::Text and set its position to the desired one
	texts.find(identifyer)->second.setPosition(position);
}

void TextRenderer::update(string identifyer, TextRenderer::MODIFY_TYPE param, string change) {
	sf::Text* toModify = &texts.find(identifyer)->second;
	switch (param) {
		case TEXT:
			toModify->setString(change);
			break;
		case FONT:
			toModify->setFont(*getFont(change));
			break;
		case SIZE:
			//Set texts size to one supplied in change <-- BAD Implementation (Template or auto caused untraceable compiler error)
			toModify->setCharacterSize(stoi(change));
			break;
		case POS:
			//Set texts position to one supplied in change <-- BAD Implementation (Template or auto caused untraceable compiler error)
			toModify->setPosition(sf::Vector2f(stoi(change.substr(0, change.find(','))), stoi(change.substr(change.find(','), change.length())) ));
			break;
	}
};

void TextRenderer::render(sf::RenderWindow& window) {
	window.setView(window.getDefaultView());
	for (auto& t : titles) {
		//Draw each text object from the map
		window.draw(texts.find(t)->second);
	}
};

void TextRenderer::loadFonts() {
	for (const auto& f : fs::directory_iterator(dir)) {
		sf::Font* font = new sf::Font(); //New font pointer
		cout << "Loading: " << f.path().string();
		if (font->loadFromFile(f.path().string())) { //Load from fonts directory
			//Insert font(s) into map
			fonts.insert({ f.path().filename().string().substr(0, f.path().filename().string().find('.')), font });
			cout << " ~Loaded! Name: " << f.path().filename().string().substr(0, f.path().filename().string().find('.'));
			fontsAvailable = true;
		}
		else {
			cout << " ~Failed!";
		}
		cout << endl;
	}
}

sf::Font* TextRenderer::getFont(string fontName) {
	if (!fonts.contains(fontName)) { //Check if font exists
		cout << "NO FONT EXISTS WITH NAME " << fontName << endl;
		exit(-5335);
	};
	sf::Font* found = fonts.find(fontName)->second;
	//Return font found in the fonts map
	return found;
}