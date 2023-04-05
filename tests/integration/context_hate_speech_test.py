import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="context_hate_speech", lang="es")


def test_racism(analyzer_es):
    output = analyzer_es.predict(
        "Hay que matarlos a todos!!!!",
        context="China prohíbe la venta de perros y gatos en los mercados")

    assert "RACISM" in output.output


def test_calls_to_action(analyzer_es):
    output = analyzer_es.predict(
        "Hay que matarlos a todos!!!!",
        context="China prohíbe la venta de perros y gatos en los mercados")

    assert "CALLS" in output.output


def test_no_hate_speech(analyzer_es):
    output = analyzer_es.predict(
        "Está perfecto, basta de crueldad animal",
        context="China prohíbe la venta de perros y gatos en los mercados")

    assert output.output == []


def test_class_hate_speech(analyzer_es):
    output = analyzer_es.predict(
        "Hay que matarlos a todos!!!!",
        context="Marcha de piqueteros por las calles de Buenos Aires")

    assert "CLASS" in output.output
    assert "CALLS" in output.output


def test_lgbti_hate_speech(analyzer_es):
    output = analyzer_es.predict(
        "Qué asco estos trolos 🤮🤮🤮🤮",
        context="Luis Novaresio y su novio se casaron en la intimidad")
    assert "LGBTI" in output.output
