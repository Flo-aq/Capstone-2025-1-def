from Functions.AuxFunctions import load_json
from os.path import join
from Functions.TranslatorFunctions import is_ordinal, next_number_is_valid, is_digit, is_special_digit, is_letter, next_char_exists, prev_char_exists, char_is_final_dot, number_to_letter
from unicodedata import normalize
import time

class Translator:
    """
    Text to braille translator that supports special characters, numbers and ordinals.

    Attributes:
        dict_characters (dict): Dictionary of basic braille characters
        dict_prefixes (dict): Dictionary of braille prefixes (uppercase, numbers, etc.)
        dict_ordinal_numbers (dict): Dictionary of ordinal numbers
        result (dict): Dictionary containing translation results
        current_paragraph (str): Current paragraph being processed
        full_text (str): Complete text to translate

    Constants:
        DEFAULT_UNICODE (str): Default unicode value for characters not found
        DEFAULT_BINARY (str): Default binary value for characters not found
    """
    DEFAULT_UNICODE = "\\u283f"
    DEFAULT_BINARY = "111111"
    DIVISION_SYMBOLS = ['/', ':', '÷']

    def __init__(self):
        try:
            paths = {
                "characters": join("BrailleDicts", "Characters.json"),
                "prefixes": join("BrailleDicts", "Prefixes.json"),
                "ordinal_numbers": join("BrailleDicts", "OrdinalNumbers.json")
            }
            self.dict_characters = load_json(paths["characters"])
            self.dict_prefixes = load_json(paths["prefixes"])
            self.dict_ordinal_numbers = load_json(paths["ordinal_numbers"])
            if not all([self.dict_characters, self.dict_prefixes, self.dict_ordinal_numbers]):
                raise ValueError("Error loading dictionary files")
            
        except Exception as e:
            raise ValueError(f"Error initializing translator: {str(e)}")

        self.result = self.initialize_dict()

        self.current_paragraph_result_dict = self.initialize_dict()
        self.current_ordinal_number_result_dict = self.initialize_dict()

        self.current_paragraph = ""
        self.full_text = ""

    def initialize_dict(self):
        """
        Initializes a dictionary to store translation results.

        Returns:
            dict: Dictionary with empty lists for characters, binary and unicode
        """
        return {
            "character": [],
            "binary": [],
            "unicode": []
        }

    def add_character(self, key, dictionary, is_ordinal=False):
        """
        Adds a character to the result dictionary in unicode, binary and character format.

        Args:
            key (str): Character key in the dictionary
            dictionary (dict): Dictionary containing the character mappings
            is_ordinal (bool): Whether the character is part of an ordinal number
        """
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        if not isinstance(dictionary, dict):
            raise ValueError("Dictionary must be a dict")
        if not key or key.strip() == '':
            return
        if key == " " or key.isspace():
            key = "blanco"
        normalized_key = normalize('NFKD', key).encode('ascii', 'ignore').decode('ascii')
        if normalized_key != key:
            if key not in dictionary:
                key = normalized_key

        target_dict = self.current_ordinal_number_result_dict if is_ordinal else self.current_paragraph_result_dict

        char_data = dictionary.get(key, {})
        unicode = dictionary.get(key, {}).get("unicode", self.DEFAULT_UNICODE)
        binary = dictionary.get(key, {}).get("binary", self.DEFAULT_BINARY)
        if "unicode" not in char_data:
            return
            # print(f"Warning: Unicode not found for character '{key}' (normalized: '{normalized_key}'), using default: {self.DEFAULT_UNICODE}")
            return
        # if "binary" not in char_data:
        #     print(f"Warning: Binary not found for character '{key}' (normalized: '{normalized_key}'), using default: {self.DEFAULT_BINARY}")

        unicode.replace("\\", "")
        unicode = bytes(unicode, 'utf-8').decode('unicode_escape')
        target_dict["unicode"].append(unicode)

        if isinstance(binary[0], list):
            target_dict["binary"].extend(''.join(str(digit) for digit in b) for b in binary)
        else:
            target_dict["binary"].append(''.join(str(digit) for digit in binary))

        key = " " if key == "blanco" else key
        target_dict["character"].append(key)
        return

    def ordinal_to_braille(self, i):
        """
        Translates an ordinal number to its braille representation.

        Args:
            i (int): Current index in the text where the ordinal number starts

        Returns:
            int: New index after processing the ordinal number

        Example:
            "1st" -> [ordinal number 1 + "st"]
            "2nd" -> [ordinal number 2 + "nd"]
        """
        self.current_ordinal_number_result_dict = self.initialize_dict()

        j = i
        number = []
        while j < len(self.current_paragraph) and self.current_paragraph[j].isdigit():
            number.append(self.current_paragraph[j])
            j += 1
        number = "".join(number)

        for digit in number:
            self.add_character(digit, self.dict_ordinal_numbers, True)

        if j < len(self.current_paragraph):
            if self.current_paragraph[j] == ".":
                j += 1

            if j < len(self.current_paragraph):
                if self.current_paragraph[j:j+2] == "er":
                    self.add_character("er", self.dict_characters, True)
                    return j+1
                elif self.current_paragraph[j] in ["o", "a"]:
                    self.add_character(
                        self.current_paragraph[j], self.dict_characters, True)
                    return j
        return j-1

    def format_numbers(self, complete_number):
        """
        Formats numbers for correct braille translation.

        Args:
            complete_number (str): Number to format

        Returns:
            str: Number formatted with thousands and decimal separators

        Example:
            "1000" -> "1.000"
            "1000,5" -> "1.000,5"
            "1000/2" -> "1.000/2"
        """
        number_without_dots = ''.join(
            char for char in complete_number if char != '.')

        division_symbol = next(
            (s for s in self.DIVISION_SYMBOLS if s in number_without_dots), None)
        parts = number_without_dots.split(division_symbol) if division_symbol else [
            number_without_dots]

        formatted_parts = []
        for part in parts:
            whole_part = part.split(',')[0] if ',' in part else part
            decimal_part = ',' + part.split(',')[1] if ',' in part else ''
            digitos = list(whole_part)[::-1]
            groups = [
                ''.join(digitos[i:i+3])[::-1]
                for i in range(0, len(digitos), 3)
            ]
            formatted_part = '.'.join(groups[::-1]) + decimal_part
            formatted_parts.append(formatted_part)

        return division_symbol.join(formatted_parts) if division_symbol else formatted_parts[0]

    def translate_full_text(self, text):
        """
        Translates complete text to braille, processing paragraphs and formatting.

        Args:
            text (str): Complete text to translate

        Raises:
            ValueError: If text is empty
        """
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if not text:
            raise ValueError("Text cannot be empty")

        start_time = time.time()
        total_chars = len(text.replace('\n', ''))
        
        self.full_text = text.split('\n\n')
        for paragraph in self.full_text:
            aux_text = paragraph.replace('-\n', '')
            aux_text = aux_text.replace('\n', ' ')
            self.current_paragraph = aux_text
            self.text_to_braille()

        self.result["character"] = "\n".join(self.result["character"])
        self.result["binary"] = "\n".join(self.result["binary"])
        self.result["unicode"] = "\n".join(self.result["unicode"])
        
        end_time = time.time()
        translation_time = end_time - start_time
        chars_per_second = total_chars / translation_time
        
        print("\n--------------------")
        print("Estadísticas de traducción:")
        print(f"Caracteres traducidos: {total_chars}")
        print(f"Tiempo total: {translation_time:.2f} segundos")
        print(f"Velocidad: {chars_per_second:.2f} caracteres/segundo")
        print("--------------------\n")
    
        return self.result

    def handle_space_translating(self, i):
        """
        Handles translation of spaces to braille.

        Args:
            i (int): Current index in text

        Returns:
            int: New index after processing space
        """
        self.add_character("blanco", self.dict_characters)
        return i + 1

    def handle_ordinal_number_translating(self, i):
        """
        Handles translation of ordinal numbers to braille.

        Args:
            i (int): Current index in text where ordinal number starts

        Returns:
            int: New index after processing ordinal number

        Example:
            "1er" -> [número prefix + 1 + "er"]
            "2do" -> [número prefix + 2 + "o"]
        """
        new_i = self.ordinal_to_braille(i)
        self.add_ordinal_numbers()
        return new_i + 1

    def handle_number_translating(self, i):
        """
        Handles translation of numbers to braille, including formatting.

        Args:
            i (int): Current index in text where number starts

        Returns:
            int: New index after processing complete number

        Example:
            "1000" -> [número prefix + "1.000"]
            "1000,5" -> [número prefix + "1.000,5"]
        """
        number_start_index = i
        current_pos = i
        while next_number_is_valid(self.current_paragraph, current_pos):
            current_pos += 1
        complete_number = self.current_paragraph[number_start_index:current_pos]
        formatted_number = self.format_numbers(complete_number)
        self.add_formatted_number(formatted_number)
        return current_pos
    
    def handle_character_after_number_translating(self, i):
        """
        Handles translation of characters that appear after numbers.

        Args:
            i (int): Current index in text where character after number appears

        Returns:
            int: New index after processing character

        Example:
            "2x3" -> [2 + multiplicar + 3]
            "2a" -> [2 + interruptor_numero + a]
        """
        if (self.current_paragraph[i].lower() == "x" and next_char_exists(self.current_paragraph, i) and is_digit(self.current_paragraph, i+1)):
            self.add_character("multiplicar", self.dict_characters)
            return i + 1
        else:
            self.add_character("interruptor_numero", self.dict_prefixes)
            self.add_character(self.current_paragraph[i], self.dict_characters)
            return i + 1
    
    def handle_uppercase_translating(self, i):
        """
        Handles translation of uppercase characters and sequences.

        Args:
            i (int): Current index in text where uppercase character starts

        Returns:
            int: New index after processing uppercase sequence

        Examples:
            "A" -> [mayus + a]
            "ABC" -> [mayus + mayus + a + b + c]
            "ABC.D" -> [mayus + mayus + a + b + c + d]
        """
        uppercase_start = i
        uppercase_end = i
        
        while uppercase_end < len(self.current_paragraph) and (
            self.current_paragraph[uppercase_end].isupper() or 
            (self.current_paragraph[uppercase_end] == "." and 
            not char_is_final_dot(self.current_paragraph, uppercase_end))
        ):
            uppercase_end += 1
        
        is_multiple_uppercase = uppercase_end - uppercase_start > 1
        
        if is_multiple_uppercase:
            self.add_character("mayus", self.dict_prefixes)
            self.add_character("mayus", self.dict_prefixes)
            
            for pos in range(uppercase_start, uppercase_end):
                if self.current_paragraph[pos] != ".":
                    self.add_character(
                        self.current_paragraph[pos].lower(), 
                        self.dict_characters
                    )
            return uppercase_end
        else:
            self.add_character("mayus", self.dict_prefixes)
            self.add_character(
                self.current_paragraph[i].lower(), 
                self.dict_characters
            )
            return i + 1

    def text_to_braille(self):
        """
        Translates the current paragraph to braille representation.
    
        This method processes the current paragraph character by character,
        handling special cases like uppercase letters, numbers, and ordinal numbers.
        Updates the result dictionaries with the translated characters.
        """
        i = 0

        self.current_paragraph_result_dict = self.initialize_dict()

        while i < len(self.current_paragraph):
            if self.current_paragraph[i] == " ":
                i = self.handle_space_translating(i)
                continue
            
            if is_ordinal(self.current_paragraph, i):
                i = self.handle_ordinal_number_translating(i)
                continue

            if i > 0 and (self.current_paragraph[i-1] == ' ' or not self.current_paragraph[i-1].isdigit()) and self.current_paragraph[i].isdigit():
                i = self.handle_number_translating(i)
                continue

            if self.current_paragraph[i].isupper():
                i = self.handle_uppercase_translating(i)
                continue

            if is_letter(self.current_paragraph, i) and prev_char_exists(i) and is_digit(self.current_paragraph, i-1):
                i = self.handle_character_after_number_translating(i)
                continue
            if len(self.current_paragraph[i]) == 0:
                i += 1
                continue
            
            self.add_character(self.current_paragraph[i], self.dict_characters)
            i += 1

        self.result["character"].append(
            "".join(self.current_paragraph_result_dict["character"]))
        self.result["binary"].append(
            " ".join(self.current_paragraph_result_dict["binary"]))
        self.result["unicode"].append(
            "".join(self.current_paragraph_result_dict["unicode"]))

    def add_ordinal_numbers(self):
        """
        Adds ordinal number translation to current paragraph result.
        """
        self.add_character("numero", self.dict_prefixes)

        for key in ["character", "binary", "unicode"]:
            self.current_paragraph_result_dict[key].extend(self.current_ordinal_number_result_dict[key])

    def add_formatted_number(self, formatted_number):
        """
        Adds formatted number translation to current paragraph result.

        Args:
            formatted_number (str): Number with proper formatting (thousands/decimal separators)
        """
        self.add_character("numero", self.dict_prefixes)
        for char in formatted_number:
            if char == ',':
                self.add_character("coma_decimal", self.dict_prefixes)
            elif char == '.':
                self.add_character("separador_millares", self.dict_prefixes)
            elif char == '/':
                self.add_character("dividido", self.dict_characters)
            else:
                self.add_character(number_to_letter(char),
                                   self.dict_characters)

