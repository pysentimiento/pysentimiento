# PySentimiento: Sentiment Analysis in Spanish

A simple Transformer-based library for Sentiment Analysis in Spanish (some other languages coming soon!).

Just do `pip install pysentimiento` and start using it:

[![Test it in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ItS0-ZPXGcEeVmRmHaneX3w8eq6Vhdde?usp=sharing)

```python
from pysentimiento import SentimentAnalyzer
analyzer = SentimentAnalyzer()
analyzer.predict("Qué gran jugador es Messi")
# returns 'POS'
analyzer.predict("Esto es pésimo")
# returns 'NEG'
analyzer.predict("Qué es esto?")
# returns 'NEU'

analyzer.predict_probas("Dónde estamos?")
# returns {'NEG': 0.10235335677862167,
# 'NEU': 0.8503277897834778,
# 'POS': 0.04731876030564308}
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

1. First, download TASS 2020 data to `data/tass2020` (you have to register [here](http://tass.sepln.org/2020/?page_id=74) to download the dataset)

Labels must be placed under `data/tass2020/test1.1/labels`

2. Run script to train models

```
python bin/train.py "dccuchile/bert-base-spanish-wwm-cased" models/beto-sentiment-analysis/ --epochs 3
```

3. Upload models to Huggingface's Model Hub

## TODO:

* Upload some other models
* Train in other languages
* Write brief paper with description

## Suggestions and bugfixes

Please use the repository [issue tracker](https://github.com/finiteautomata/pysentimiento/issues) to point out bugs and make suggestions (new models, use another datasets, some other languages, etc)
