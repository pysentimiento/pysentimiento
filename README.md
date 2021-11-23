# pysentimiento: A Python toolkit for Sentiment Analysis and Social NLP tasks


![Tests](https://github.com/finiteautomata/pysentimiento/workflows/run_tests/badge.svg)

A Transformer-based library for SocialNLP classification tasks.

Currently supports:

- Sentiment Analysis (Spanish, English)
- Emotion Analysis (Spanish, English)
- Hate Speech Detection (Spanish, English)


Just do `pip install pysentimiento` and start using it:

[![Test it in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pysentimiento/pysentimiento/blob/master/notebooks/PySentimiento_Sentiment_Analysis_in_Spanish.ipynb)

```python
from pysentimiento import create_analyzer
analyzer = create_analyzer(task="sentiment", lang="es")

analyzer.predict("Qué gran jugador es Messi")
# returns AnalyzerOutput(output=POS, probas={POS: 0.998, NEG: 0.002, NEU: 0.000})
analyzer.predict("Esto es pésimo")
# returns AnalyzerOutput(output=NEG, probas={NEG: 0.999, POS: 0.001, NEU: 0.000})
analyzer.predict("Qué es esto?")
# returns AnalyzerOutput(output=NEU, probas={NEU: 0.993, NEG: 0.005, POS: 0.002})

analyzer.predict("jejeje no te creo mucho")
# AnalyzerOutput(output=NEG, probas={NEG: 0.587, NEU: 0.408, POS: 0.005})
"""
Emotion Analysis in English
"""

analyzer = create_analyzer(task="emotion", lang="en")

emotion_analyzer.predict("yayyy")
# returns AnalyzerOutput(output=joy, probas={joy: 0.723, others: 0.198, surprise: 0.038, disgust: 0.011, sadness: 0.011, fear: 0.010, anger: 0.009})
emotion_analyzer.predict("fuck off")
# returns AnalyzerOutput(output=anger, probas={anger: 0.798, surprise: 0.055, fear: 0.040, disgust: 0.036, joy: 0.028, others: 0.023, sadness: 0.019})

"""
Hate Speech (misogyny & racism)
"""
hate_speech_analyzer = create_analyzer(task="hate_speech", lang="es")

hate_speech_analyzer.predict("Esto es una mierda pero no es odio")
# returns AnalyzerOutput(output=[], probas={hateful: 0.022, targeted: 0.009, aggressive: 0.018})
hate_speech_analyzer.predict("Esto es odio porque los inmigrantes deben ser aniquilados")
# returns AnalyzerOutput(output=['hateful'], probas={hateful: 0.835, targeted: 0.008, aggressive: 0.476})

hate_speech_analyzer.predict("Vaya guarra barata y de poca monta es Miley!")
# returns AnalyzerOutput(output=['hateful', 'targeted', 'aggressive'], probas={hateful: 0.987, targeted: 0.978, aggressive: 0.969})
```

Also, you might use pretrained models directly with [`transformers`](https://github.com/huggingface/transformers) library.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("pysentimiento/robertuito-sentiment-analysis")

model = AutoModelForSequenceClassification.from_pretrained("pysentimiento/robertuito-sentiment-analysis")
```

## Preprocessing

`pysentimiento` features a tweet preprocessor specially suited for tweet classification with transformer-based models.

```python
from pysentimiento.preprocessing import preprocess_tweet

# Replaces user handles and URLs by special tokens
preprocess_tweet("@perezjotaeme debería cambiar esto http://bit.ly/sarasa") # "@usuario debería cambiar esto url"

# Shortens repeated characters
preprocess_tweet("no entiendo naaaaaaaadaaaaaaaa", shorten=2) # "no entiendo naadaa"

# Normalizes laughters
preprocess_tweet("jajajajaajjajaajajaja no lo puedo creer ajajaj") # "jaja no lo puedo creer jaja"

# Handles hashtags
preprocess_tweet("esto es #UnaGenialidad")
# "esto es una genialidad"

# Handles emojis
preprocess_tweet("🎉🎉", lang="en")
# 'emoji party popper emoji emoji party popper emoji'
```

## Trained models so far

Check [CLASSIFIERS.md](CLASSIFIERS.md) for details on the reported performances of each model.


## Instructions for developers

0. Clone and install

```
git clone https://github.com/pysentimiento/pysentimiento
pip install poetry
poetry shell
poetry install
```

1. Download data TASS 2020 data to `data/tass2020` (you have to register [here](http://tass.sepln.org/2020/?page_id=74) to download the dataset)

Labels must be placed under `data/tass2020/test1.1/labels`

Open an issue or email us if you are not able to get the data.

2. Run script to train models

Check [TRAIN.md](TRAIN.md) for further information on how to train your models


3. Upload models to Huggingface's Model Hub

Check ["Model sharing and upload"](https://huggingface.co/transformers/model_sharing.html) instructions in `huggingface` docs.

## License

`pysentimiento` is an open-source library. However, please be aware that models are trained with third-party datasets and are subject to their respective licenses, many of which are for non-commercial use

1. [TASS Dataset license](http://tass.sepln.org/tass_data/download.php) (License for Sentiment Analysis in Spanish, Emotion Analysis in Spanish & English)
2. [SEMEval 2017 Dataset license](https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip?file_subpath=%2F2017_English_final%2FDOWNLOAD%2FREADME.txt) (Sentiment Analysis in English)

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


## Suggestions and bugfixes

Please use the repository [issue tracker](https://github.com/finiteautomata/pysentimiento/issues) to point out bugs and make suggestions (new models, use another datasets, some other languages, etc)
