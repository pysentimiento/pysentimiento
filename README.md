# pysentimiento: A Python toolkit for Sentiment Analysis and Social NLP tasks


![Tests](https://github.com/finiteautomata/pysentimiento/workflows/run_tests/badge.svg)

A Transformer-based library for SocialNLP classification tasks.

Currently supports:

- Sentiment Analysis (Spanish, English)
- Emotion Analysis (Spanish, English)


Just do `pip install pysentimiento` and start using it:

[![Test it in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/finiteautomata/pysentimiento/blob/master/notebooks/PySentimiento_Sentiment_Analysis_in_Spanish.ipynb)

```python
from pysentimiento import SentimentAnalyzer
analyzer = SentimentAnalyzer(lang="es")

analyzer.predict("Qué gran jugador es Messi")
# returns SentimentOutput(output=POS, probas={POS: 0.998, NEG: 0.002, NEU: 0.000})
analyzer.predict("Esto es pésimo")
# returns SentimentOutput(output=NEG, probas={NEG: 0.999, POS: 0.001, NEU: 0.000})
analyzer.predict("Qué es esto?")
# returns SentimentOutput(output=NEU, probas={NEU: 0.993, NEG: 0.005, POS: 0.002})

analyzer.predict("jejeje no te creo mucho")
# SentimentOutput(output=NEG, probas={NEG: 0.587, NEU: 0.408, POS: 0.005})
"""
Emotion Analysis in English
"""

emotion_analyzer = EmotionAnalyzer(lang="en")

emotion_analyzer.predict("yayyy")
# returns EmotionOutput(output=joy, probas={joy: 0.723, others: 0.198, surprise: 0.038, disgust: 0.011, sadness: 0.011, fear: 0.010, anger: 0.009})
emotion_analyzer.predict("fuck off")
# returns EmotionOutput(output=anger, probas={anger: 0.798, surprise: 0.055, fear: 0.040, disgust: 0.036, joy: 0.028, others: 0.023, sadness: 0.019})

```

Also, you might use pretrained models directly with [`transformers`](https://github.com/huggingface/transformers) library.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("finiteautomata/beto-sentiment-analysis")

model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/beto-sentiment-analysis")
```

## Preprocessing

`pysentimiento` features a tweet preprocessor specially suited for tweet classification with transformer-based models.

```python
from pysentimiento.preprocessing import preprocess_tweet

# Replaces user handles and URLs by special tokens
preprocess_tweet("@perezjotaeme debería cambiar esto http://bit.ly/sarasa") # "[USER] debería cambiar esto [URL]"

# Shortens repeated characters
preprocess_tweet("no entiendo naaaaaaaadaaaaaaaa", shorten=2) # "no entiendo naadaa"

# Normalizes laughters
preprocess_tweet("jajajajaajjajaajajaja no lo puedo creer ajajaj") # "jaja no lo puedo creer jaja"

# Handles hashtags
preprocess_tweet("esto es #UnaGenialidad")
# "esto es una genialidad"

# Handles emojis
preprocess_tweet("🎉🎉", lang="en")
# '[EMOJI] party popper [EMOJI][EMOJI] party popper [EMOJI]'
```

## Trained models so far

Check [CLASSIFIERS.md](CLASSIFIERS.md) for details on the reported performances of each model.

### Spanish models

- [`beto-sentiment-analysis`](https://huggingface.co/finiteautomata/beto-sentiment-analysis)
- [`beto-emotion-analysis`](https://huggingface.co/finiteautomata/beto-emotion-analysis)

### English models

- [`bertweet-base-sentiment-analysis`](https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis)
- [`bertweet-base-emotion-analysis`](https://huggingface.co/finiteautomata/bertweet-base-emotion-analysis)


## Instructions for developers

1. First, download TASS 2020 data to `data/tass2020` (you have to register [here](http://tass.sepln.org/2020/?page_id=74) to download the dataset)

Labels must be placed under `data/tass2020/test1.1/labels`

2. Run script to train models

Check [TRAIN_EVALUATE.md](TRAIN_EVALUATE.md)

3. Upload models to Huggingface's Model Hub

Check ["Model sharing and upload"](https://huggingface.co/transformers/model_sharing.html) instructions in `huggingface` docs.

## Citation

If you use `pysentimiento` in your work, please cite [this paper](https://arxiv.org/abs/2106.09462)

```
@misc{perez2021pysentimiento,
      title={pysentimiento: A Python Toolkit for Sentiment Analysis and SocialNLP tasks}, 
      author={Juan Manuel Pérez and Juan Carlos Giudici and Franco Luque},
      year={2021},
      eprint={2106.09462},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## TODO:

* Upload some other models
* Train in other languages

## Suggestions and bugfixes

Please use the repository [issue tracker](https://github.com/finiteautomata/pysentimiento/issues) to point out bugs and make suggestions (new models, use another datasets, some other languages, etc)
