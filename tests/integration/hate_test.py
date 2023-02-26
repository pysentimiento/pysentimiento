import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="hate_speech", lang="es")


@pytest.fixture
def analyzer_en():
    return create_analyzer(task="hate_speech", lang="en")


@pytest.fixture
def analyzer_it():
    return create_analyzer(task="hate_speech", lang="it")


def test_not_hateful(analyzer_es):
    assert analyzer_es.predict("Qué buena onda!!!!").output == []


def test_hateful(analyzer_es):
    assert "hateful" in analyzer_es.predict(
        "Odio a todos los putos negros de mierda!").output


def test_not_hateful_en(analyzer_en):
    assert analyzer_en.predict("Oh man that's really cool!").output == []


def test_hateful_en(analyzer_en):
    assert "hateful" in analyzer_en.predict(
        "I think niggers should be killed").output


def test_hateful_it(analyzer_it):
    assert "hateful" in analyzer_it.predict(
        "Penso che i negri dovrebbero essere uccisi").output


def test_stereotype_it(analyzer_it):
    assert "stereotype" in analyzer_it.predict(
        "Tutti i musulmani sono terroristi").output
