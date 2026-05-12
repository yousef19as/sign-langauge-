"""Arabic sign language letter mappings."""

ARABIC_LETTERS_MAP: dict[str, str] = {
    "aleff": "ا", "bb": "ب", "taa": "ت", "thaa": "ث",
    "jeem": "ج", "haa": "ح", "khaa": "خ", "dal": "د",
    "thal": "ذ", "ra": "ر", "zay": "ز", "seen": "س",
    "sheen": "ش", "saad": "ص", "dhad": "ض", "ta": "ط",
    "dha": "ظ", "ain": "ع", "ghain": "غ", "fa": "ف",
    "gaaf": "ق", "kaaf": "ك", "laam": "ل", "meem": "م",
    "nun": "ن", "ha": "ه", "waw": "و", "ya": "ئ",
    "toot": "ة", "al": "ال", "la": "لا", "yaa": "ي",
}

ARABIC_CLASS_NAMES: list[str] = sorted(ARABIC_LETTERS_MAP.keys())

# ASL classes — 29 classes (A-Z + del, nothing, space)
ASL_CLASS_NAMES: list[str] = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]
