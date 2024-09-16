import re
import sys
import unicodedata


def fix_hyphens(arg: str):
    return re.sub(r"^--([^=]+)", lambda m: "--" + m.group(1).replace("-", "_"), arg)

def sanitize_name(name: str) -> str:
    """
    Sanitize a string to be safely used as a valid filename, folder name, or Python module name.

    Parameters:
        name (str): The input string to sanitize.

    Returns:
        str: The sanitized name.
    """
    # Remove leading/trailing whitespace and normalize unicode characters
    name = unicodedata.normalize('NFKC', name.strip())

    # Define invalid characters based on the operating system
    if sys.platform.startswith('win'):
        invalid_chars = r'[<>:"/\\|?*\x00-\x1F]'
        reserved_names = {
            'CON', 'PRN', 'AUX', 'NUL',
            *(f'COM{i}' for i in range(1, 10)),
            *(f'LPT{i}' for i in range(1, 10)),
        }
    else:
        invalid_chars = r'[<>:"/\\|?*\x00]'
        reserved_names = set()

    # Replace invalid characters with underscores
    name = re.sub(invalid_chars, '_', name)

    # Remove trailing periods and spaces (Windows limitation)
    name = name.rstrip('. ')

    # Handle reserved device names (Windows)
    name_part, dot, extension = name.partition('.')
    if name_part.upper() in reserved_names:
        name_part = f'_{name_part}'
    name = name_part + (dot + extension if dot else '')

    # Ensure the name is a valid Python identifier (module name)
    # Replace invalid characters with underscores
    name = re.sub(r'\W|^(?=\d)', '_', name)
    # Ensure it doesn't start with a digit
    if not re.match(r'[A-Za-z_]', name):
        name = f'_{name}'
    # Remove any leading or trailing underscores
    name = name.strip('_')

    # If the result is empty, return a default name
    if not name:
        name = 'default_name'

    return name