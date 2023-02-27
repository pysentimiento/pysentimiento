import pytest
from pysentimiento import create_analyzer


@pytest.fixture
def analyzer_es():
    return create_analyzer(task="sentiment", lang="es")


@pytest.fixture
def analyzer_en():
    return create_analyzer(task="sentiment", lang="en")


@pytest.fixture
def analyzer_it():
    return create_analyzer(task="sentiment", lang="it")


@pytest.fixture
def analyzer_pt():
    return create_analyzer(task="sentiment", lang="pt")


def test_analyze_negative_sent(analyzer_es):
    assert analyzer_es.predict("Esto es pésimo").output == "NEG"


def test_analyze_neu_sent(analyzer_es):
    assert analyzer_es.predict("Qué es esto?").output == "NEU"


def test_multiple_sentences(analyzer_es):
    sentences = [
        "Esto es muy bueno!",
        "Esto es pésimo",
        "Qué es esto?"
    ] * 2

    ret = analyzer_es.predict(sentences)

    outputs = [r.output for r in ret]

    assert outputs == ["POS", "NEG", "NEU", "POS", "NEG", "NEU"]


def test_analyze_negative_sent_en(analyzer_en):
    assert analyzer_en.predict("This is shitty").output == "NEG"


def test_analyze_neu_sent_en(analyzer_en):
    assert analyzer_en.predict("What's this?").output == "NEU"


def test_multiple_sentences_en(analyzer_en):
    sentences = [
        "omg this is wonderful!",
        "This is shitty",
        "What's this?"
    ] * 2

    ret = analyzer_en.predict(sentences)

    outputs = [r.output for r in ret]

    assert outputs == ["POS", "NEG", "NEU", "POS", "NEG", "NEU"]


def test_analyze_negative_sent_it(analyzer_it):
    assert analyzer_it.predict("Questo è pessimo").output == ["neg"]


def test_analyze_neu_sent_it(analyzer_it):
    assert analyzer_it.predict("Cosa è questo?").output == []


def test_analyze_pos_sent_it(analyzer_it):
    assert analyzer_it.predict("Questo è fantastico").output == ["pos"]


def test_analyze_negative_sent_pt(analyzer_pt):
    assert analyzer_pt.predict("Isso é péssimo").output == "NEG"


def test_analyze_neu_sent_pt(analyzer_pt):
    # Translate "Esta es una noticia acerca del clima"
    assert analyzer_pt.predict(
        "Esta é uma notícia sobre o clima").output == "NEU"
