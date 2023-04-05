import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="targeted_sentiment", lang="es")


def test_negative_target(analyzer_es):
    output = analyzer_es.predict(
        "Alberto Fernández durante un acto en Hurlingham: \"El gobierno de Macri fue un desastre\"",
        target="Macri")

    assert output.output == "NEG"


def test_positive_target(analyzer_es):
    output = analyzer_es.predict(
        "Alberto Fernández durante un acto en Hurlingham: \"El gobierno de Macri fue un desastre\"",
        target="Alberto Fernández")

    assert output.output == "POS"
