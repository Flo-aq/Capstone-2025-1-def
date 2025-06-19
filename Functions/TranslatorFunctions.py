def next_number_is_valid(text, i):
    """
    Checks if the next character is a valid part of the current number.

    Args:
        text (str): Text being processed
        i (int): Current position in text

    Returns:
        bool: True if next character is valid part of number, False otherwise

    Example:
        "123.456" -> True for positions 0-6
        "123." -> False (final dot)
        "123,456" -> True
    """
    if char_exists(text, i):
        if is_digit(text, i):
            return True
        if is_special_digit(text, i) and not char_is_final_dot(text, i) and not char_is_separation_comma(text, i):
            if next_char_exists(text, i):
                if text[i+1] in "." and char_is_final_dot(text, i+1):
                    return True
                elif is_digit(text, i+1):
                    return True
                else:
                    return False
            else:
                return False
    return False


def is_digit(text, i):
    return text[i].isdigit()


def is_special_digit(text, i):
    return text[i] in "./:÷,"


def is_letter(text, i):
    return text[i].lower() in "abcdefghijklmnñopqrstuvwxyz"


def char_exists(text, i):
    return i < len(text)


def next_char_exists(text, i):
    return i + 1 < len(text)


def prev_char_exists(i):
    return i - 1 >= 0


def char_is_final_dot(text, i):
    return text[i] == '.' and ((next_char_exists(text, i) and text[i+1] == ' ') or not next_char_exists(text, i))


def char_is_separation_comma(text, i):
    return text[i] == ',' and ((next_char_exists(text, i) and (text[i+1] == ' ' or not is_digit(text, i+1))) or not next_char_exists(text, i))


def is_ordinal(text, i):
    """
    Checks if number starting at position i is an ordinal number.

    Args:
        text (str): Text being processed
        i (int): Starting position

    Returns:
        bool: True if number is ordinal, False otherwise

    Example:
        "1er" -> True
        "2o" -> True
        "3a" -> True
        "123" -> False
    """
    if not char_exists(text, i):
        return False

    j = i
    while j < len(text) and text[j].isdigit():
        j += 1

    if j == i:
        return False

    if j < len(text):
        if text[j] == "." and j + 1 < len(text) and text[j+1] in ["o", "a"]:
            return True

        if text[j] in ["o", "a"]:
            return True

        if text[j-1] in ["1", "3"]: 
            if text[j] == "." and j + 2 < len(text) and text[j+1:j+3] == "er":
                return True
            elif text[j:j+2] == "er":
                return True
    return False


def get_ordinal_length(text, i):
    """
    Gets the total length of an ordinal number.

    Args:
        text (str): Text being processed
        i (int): Starting position of ordinal

    Returns:
        int: Total length of ordinal number including suffix

    Example:
        "1er" -> 3
        "2o" -> 2
        "3a" -> 2
    """
    j = i

    while j < len(text) and text[j].isdigit():
        j += 1

    if j < len(text):
        if text[j] == "." and j + 1 < len(text) and text[j+1] in ["o", "a"]:
            return j - i + 2
        if text[j] in ["o", "a"]:
            return j - i + 1
        if text[j-1] in ["1", "3"]:
            if text[j] == "." and j + 2 < len(text) and text[j+1:j+3] == "er":
                return j - i + 3
            elif text[j:j+2] == "er":
                return j - i + 2

    return j - i


def number_to_letter(number):
    """
    Converts a number to its corresponding letter (1=a, 2=b, etc).

    Args:
        number (str): Number to convert

    Returns:
        str: Corresponding letter (0='j', 1='a', 2='b', etc)

    Example:
        "1" -> "a"
        "0" -> "j"
        "3" -> "c"
    """
    if number == "0":
        return "j"
    else:
        letter_idx = int(number) - 1
        letter = chr(ord('a') + letter_idx)
        return letter
