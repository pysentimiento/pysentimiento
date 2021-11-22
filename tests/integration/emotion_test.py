import pytest
from pysentimiento import SentimentAnalyzer
from pysentimiento.analyzer import EmotionAnalyzer


@pytest.fixture
def analyzer_es():
    return EmotionAnalyzer(lang="es")

@pytest.fixture
def analyzer_en():
    return EmotionAnalyzer(lang="en")


def test_anger(analyzer_es):
    assert analyzer_es.predict("Tengo ganas de matarlos a todos estos hd putas").output == "anger"


def test_joy(analyzer_es):
    assert analyzer_es.predict("Ohhh qué lindo que es todo esto!").output == "joy"




def test_anger_en(analyzer_en):
    assert analyzer_en.predict("I'm pissed off, I hate this shit").output == "anger"


def test_joy_en(analyzer_en):
    assert analyzer_en.predict("This is quite nice!").output == "joy"