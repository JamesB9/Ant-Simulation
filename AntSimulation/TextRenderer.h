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

	///Types of modification - See Update Function
	const enum MODIFY_TYPE {
		TEXT, FONT, SIZE, POS
	};

	////////////////////////////////////////////////////////////
	/// \brief Write a string to the screen
	/// 
	/// \param string identifyer: unique ID for the string
	/// \param string text: Text to be rendered on screen
	/// \param int size: Size of text to be rendered on screen 
	/// \param Vector2f position: position where text should be rendered
	/// \param font: Default if blank, chosen font to be used when rendering this text
	/// 
	////////////////////////////////////////////////////////////
	void write(string identifyer, string text, int size, sf::Vector2f position, string font = "Default");

	////////////////////////////////////////////////////////////
	/// \brief Update a existing screen text
	///
	/// Note that we tries using Template functions or auto for
	/// the parameter change, however it caused an untraceable
	/// compiler error.
	///
	/// \param identifyer unique ID for the text
	/// \param param part of the sf::Text you want to edit
	/// \param change String instance of change wanting to be
	/// 
	/// \see MODIFY_TYPE
	////////////////////////////////////////////////////////////
	void update(string identifyer, TextRenderer::MODIFY_TYPE param, string change);

	////////////////////////////////////////////////////////////
	/// \brief Write a string to the screen
	/// 
	/// \param RenderWindow window - Renders the text on the desired window
	/// 
	////////////////////////////////////////////////////////////
	void render(sf::RenderWindow& window);

	////////////////////////////////////////////////////////////
	/// \brief Constructor for a TextRenderer
	////////////////////////////////////////////////////////////
	TextRenderer();


private:

	///Signature for fonts being available, False = no text allowed on screen
	bool fontsAvailable = false;
	///Fixed directory of fonts
	string dir = "./Fonts/";
	///Vector of strings containing each font's title
	vector<string> titles;
	///Map where a unique key points towards a sf::Text object
	map<string, sf::Text> texts;
	///Map where a unique key points towards a font pointer
	map<string, sf::Font*> fonts;
	
	////////////////////////////////////////////////////////////
	/// \brief Load fonts from fixed directory
	/// 
	/// \see dir
	////////////////////////////////////////////////////////////
	void loadFonts();

	////////////////////////////////////////////////////////////
	/// \brief Get a font via its string name
	/// 
	/// \param string fontName: unique ID for the font
	/// 
	/// \return Font Pointer
	/// 
	////////////////////////////////////////////////////////////
	sf::Font* getFont(string fontName);
};