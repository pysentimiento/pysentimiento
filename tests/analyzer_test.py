from pysentimiento import SentimentAnalyzer


analyzer = SentimentAnalyzer(lang="es")

def test_analyze_negative_sent():
    assert analyzer.predict("Esto es pésimo").output == "NEG"

def test_analyze_neu_sent():
    assert analyzer.predict("Qué es esto?").output == "NEU"