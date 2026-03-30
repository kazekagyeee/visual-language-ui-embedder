"""
text_preprocessing.py
---------------------
Текстовая предобработка перед передачей в эмбеддер:
  1. Нижний регистр
  2. Удаление всего кроме букв и пробелов
  3. Удаление стоп-слов (русских и английских)
  4. Лемматизация (pymorphy3 для русского, WordNetLemmatizer для английского)
  5. Сборка в строку

Использование:
    from text_preprocessing import preprocess_text
    clean = preprocess_text("Быстрые лисы бегут через лес!")
"""

import re
import unicodedata
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK ресурсы (скачиваются один раз при первом импорте)
# ---------------------------------------------------------------------------
import nltk

def _ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt",          "punkt"),
        ("tokenizers/punkt_tab",      "punkt_tab"),
        ("corpora/stopwords",         "stopwords"),
        ("corpora/wordnet",           "wordnet"),
        ("corpora/omw-1.4",           "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)

_ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# pymorphy3 (русская лемматизация)
# ---------------------------------------------------------------------------
try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
    _HAS_PYMORPHY = True
except ImportError:
    logger.warning(
        "pymorphy3 не установлен — русские слова не будут лемматизированы. "
        "Установите: pip install pymorphy3"
    )
    _morph = None
    _HAS_PYMORPHY = False

# ---------------------------------------------------------------------------
# Вспомогательные ресурсы
# ---------------------------------------------------------------------------
_stop_ru = set(stopwords.words("russian"))
_stop_en = set(stopwords.words("english"))
_STOP_WORDS = _stop_ru | _stop_en

_wnl = WordNetLemmatizer()

_RE_CLEAN = re.compile(r"[^a-zа-яёA-ZА-ЯЁ\s]")
_RE_SPACES = re.compile(r"\s+")


def _detect_lang(word: str) -> str:
    """Возвращает 'ru' если слово содержит кириллицу, иначе 'en'."""
    for ch in word:
        if unicodedata.category(ch) in ("Ll", "Lu"):
            name = unicodedata.name(ch, "")
            if "CYRILLIC" in name:
                return "ru"
    return "en"


@lru_cache(maxsize=16384)
def _lemmatize(word: str) -> str:
    """Лемматизирует одно слово (с кешем для ускорения)."""
    lang = _detect_lang(word)
    if lang == "ru":
        if _HAS_PYMORPHY:
            return _morph.parse(word)[0].normal_form
        return word
    else:
        # WordNetLemmatizer работает только с lowercase-английским
        return _wnl.lemmatize(word.lower())


def preprocess_text(text: str) -> str:
    """
    Выполняет полный цикл предобработки текста:
      - нижний регистр
      - удаление спецсимволов и цифр
      - токенизация
      - удаление стоп-слов
      - лемматизация (русский + английский)

    Возвращает строку с пробелами в качестве разделителя.
    Безопасно работает с пустыми строками и None.
    """
    if not text:
        return ""

    # 1. Нижний регистр
    text = text.lower()

    # 2. Убираем всё кроме букв и пробелов
    text = _RE_CLEAN.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()

    if not text:
        return ""

    # 3. Токенизация
    try:
        tokens = word_tokenize(text, language="russian")
    except Exception:
        tokens = text.split()

    # 4. Стоп-слова + лемматизация
    result_tokens = []
    for token in tokens:
        if not token.isalpha():          # пропускаем не-буквенные фрагменты
            continue
        if token in _STOP_WORDS:         # фильтр стоп-слов
            continue
        lemma = _lemmatize(token)
        if lemma and lemma not in _STOP_WORDS:
            result_tokens.append(lemma)

    return " ".join(result_tokens)
