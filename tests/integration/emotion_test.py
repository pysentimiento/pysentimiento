import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="emotion", lang="es")


@pytest.fixture
def analyzer_en():
    return create_analyzer(task="emotion", lang="en")


@pytest.fixture
def analyzer_it():
    return create_analyzer(task="emotion", lang="it")


@pytest.fixture
def analyzer_pt():
    return create_analyzer(task="emotion", lang="pt")


def test_anger(analyzer_es):
    assert (
        analyzer_es.predict("Tengo ganas de matarlos a todos estos hd putas").output
        == "anger"
    )


def test_joy(analyzer_es):
    assert analyzer_es.predict("Ohhh qué lindo que es todo esto!").output == "joy"


def test_anger_en(analyzer_en):
    assert analyzer_en.predict("I'm pissed off, I hate this shit").output == "anger"


def test_joy_en(analyzer_en):
    assert analyzer_en.predict("This is quite nice!").output == "joy"


def test_anger_it(analyzer_it):
    assert analyzer_it.predict("Sono arrabbiato, odio questa merda").output == "anger"


def test_anger_pt(analyzer_pt):
    assert analyzer_pt.predict("Eles são todos uma merda, eu os odeio").output == [
        "anger"
    ]
