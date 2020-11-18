# PySentimiento: Sentiment Analysis in Spanish

A simple Transformer-based library for Spanish.



```python
from pysentimiento import SentimentAnalyzer
analyzer = SentimentAnalyzer()
analyzer.predict("Qué gran jugador es Messi")
# returns 'POS'
analyzer.predict("Esto es pésimo")
# returns 'NEG'
analyzer.predict("Qué es esto?")
# returns 'NEU'
analyzer.predict_probas("Qué es esta cosa?")

# returns {'NEG': 0.7448181509971619,
# 'NEU': 0.22246581315994263,
# 'POS': 0.032716117799282074}
```

Also, you might use pretrained models directly with [`transformers`](https://github.com/huggingface/transformers) library.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")

model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/beto-sentiment-analysis")
```

## Trained models so far

- [`beto-sentiment-analysis`](https://huggingface.co/finiteautomata/beto-sentiment-analysis)

## Instructions for developers

1. First, download TASS 2020 data to `data/tass2020`
2. Run notebooks to train models
3. Upload models to Huggingface's Model Hub

## TODO:

* Upload some other models
* Train in other languages
* Write brief paper with description
