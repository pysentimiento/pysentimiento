import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="ner", lang="es")


def test_analyze_sent_without_entities(analyzer_es):
    assert analyzer_es.predict("Esto es pésimo") == []


def test_analyze_sent_with_location(analyzer_es):
    sentence = "Estamos en Argentina"
    entities = analyzer_es.predict(sentence)

    assert len(entities) == 1
    assert entities[0]["text"] == "Argentina"
    assert entities[0]["type"] == "LOC"
