import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="irony", lang="es")


@pytest.fixture
def analyzer_en():
    return create_analyzer(task="irony", lang="en")


@pytest.fixture
def analyzer_it():
    return create_analyzer(task="irony", lang="it")


def test_analyze_not_ironic_es(analyzer_es):
    assert analyzer_es.predict("Esto es muy bueno").output == "not ironic"


def test_analyze_ironic_es(analyzer_es):
    assert analyzer_es.predict(
        "Ah, en serio? Mirá, te juro que no sabía, gracias por iluminarme").output == "ironic"


def test_analyze_not_ironic_en(analyzer_en):
    assert analyzer_en.predict(
        "I don't really think so").output == "not ironic"


def test_analyze_ironic_en(analyzer_en):
    assert analyzer_en.predict(
        "Oh, really? Look, Believe I didn't know, thanks for enlightning me").output == "ironic"


def test_analyze_not_ironic_it(analyzer_it):
    assert analyzer_it.predict(
        "Questo è molto bono").output == "not ironic"


def test_analyze_ironic_it(analyzer_it):
    assert analyzer_it.predict(
        "formato il nuovo governo Monti. Che Dio ce la mandi tecnicamente buona").output == "ironic"
