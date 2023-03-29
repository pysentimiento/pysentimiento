import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def ner_analyzer_es():
    return create_analyzer(task="ner", lang="es")


@pytest.fixture
def ner_analyzer_en():
    return create_analyzer(task="ner", lang="en")


@pytest.fixture
def pos_analyzer_es():
    return create_analyzer(task="pos", lang="es")


@pytest.fixture
def pos_analyzer_en():
    return create_analyzer(task="pos", lang="en")


def test_analyze_sent_without_entities(ner_analyzer_es):
    assert ner_analyzer_es.predict("Esto es pésimo").entities == []


def test_analyze_sent_with_location(ner_analyzer_es):
    sentence = "Estamos en Argentina"
    entities = ner_analyzer_es.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "Argentina"
    assert entities[0]["type"] == "LOC"


def test_en_analyze_sent_with_proper_name(ner_analyzer_en):
    sentence = "My name is John Brown"
    entities = ner_analyzer_en.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "John Brown"
    assert entities[0]["type"] == "PER"


def test_es_analyze_sent_with_user_name(ner_analyzer_es):
    sentence = "Mi nombre es @johndoe"
    entities = ner_analyzer_es.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "@johndoe"
    assert entities[0]["type"] == "USER"


def test_en_analyze_sent_with_user_name(ner_analyzer_en):
    sentence = "I hate @realDonaldTrump"
    entities = ner_analyzer_en.predict(sentence).entities

    assert len(entities) == 1
    assert entities[0]["text"] == "@realDonaldTrump"
    assert entities[0]["type"] == "USER"


def test_en_pos_analyze_sent_with_user_name(pos_analyzer_en):
    sentence = "I hate @realDonaldTrump"
    output = pos_analyzer_en.predict(sentence)

    assert len(output.labels) == 3
    assert output.labels[0] == "PRON"
    assert output.labels[1] == "VERB"
    assert output.labels[2] == "USER"


def test_es_pos_analyze_sent_with_user_name(pos_analyzer_es):
    sentence = "Mi nombre es @johndoe"
    output = pos_analyzer_es.predict(sentence)

    assert len(output.labels) == 4
    assert output.labels[0] == "DET"
    assert output.labels[1] == "NOUN"
    assert output.labels[2] == "VERB"
    assert output.labels[3] == "USER"
