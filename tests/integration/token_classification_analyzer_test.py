import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="ner", lang="es")


@pytest.fixture
def analyzer_en():
    return create_analyzer(task="ner", lang="en")


def test_analyze_sent_without_entities(analyzer_es):
    assert analyzer_es.predict("Esto es pésimo").entities == []


def test_analyze_sent_with_location(analyzer_es):
    sentence = "Estamos en Argentina"
    entities = analyzer_es.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "Argentina"
    assert entities[0]["type"] == "LOC"


def test_en_analyze_sent_with_proper_name(analyzer_en):
    sentence = "My name is John Brown"
    entities = analyzer_en.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "John Brown"
    assert entities[0]["type"] == "PER"


def test_es_analyze_sent_with_user_name(analyzer_es):
    sentence = "Mi nombre es @johndoe"
    entities = analyzer_es.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "@johndoe"
    assert entities[0]["type"] == "USER"


def test_en_analyze_sent_with_user_name(analyzer_es):
    sentence = "I hate @realDonaldTrump"
    entities = analyzer_es.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "@realDonaldTrump"
    assert entities[0]["type"] == "USER"
