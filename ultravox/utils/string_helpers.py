import re

def fix_hyphens(arg: str):
    return re.sub(r"^--([^=]+)", lambda m: "--" + m.group(1).replace("-", "_"), arg)

def normalize_filename(filename: str):
    return re.sub(r'[^\w\-_\.]', '_', filename)