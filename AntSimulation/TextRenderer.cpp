#include "TextRenderer.h";


TextRenderer::TextRenderer() {
	loadFonts();
}

void TextRenderer::write(string identifyer, string text, int size, sf::Vector2f position, string font) {
	if (!fontsAvailable) { printf("NO FONTS AVAILABLE, stopped text rendering\n"); return; }
	titles.push_back(identifyer);
	texts.insert({ identifyer, sf::Text(text, *getFont(font), size) });
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
			//toModify->setCharacterSize(change);
			break;
		case POS:
			//toModify->setPosition(change);
			break;
	}
};

void TextRenderer::render(sf::RenderWindow& window) {
	window.setView(window.getDefaultView());
	for (auto& t : titles) {
		window.draw(texts.find(t)->second);
	}
};

void TextRenderer::loadFonts() {
	for (const auto& f : fs::directory_iterator(dir)) {
		sf::Font* font = new sf::Font();
		cout << "Loading: " << f.path().string();
		if (font->loadFromFile(f.path().string())) {
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
	if (!fonts.contains(fontName)) {
		cout << "NO FONT EXISTS WITH NAME " << fontName << endl;
		exit(-5335);
	};
	sf::Font* found = fonts.find(fontName)->second;
	return found;
}