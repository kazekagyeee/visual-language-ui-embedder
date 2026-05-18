# -*- coding: utf-8 -*-

def split_query_to_ui_phrases(query, known_phrases=None, max_phrases=1):
    """
    Не дробим пользовательский вопрос на отдельные слова.
    Один вопрос = один semantic UI-запрос.
    """
    return [query]
