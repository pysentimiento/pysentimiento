# pysentimiento: A Python toolkit for Sentiment Analysis and Social NLP tasks

![Tests](https://github.com/finiteautomata/pysentimiento/workflows/run_tests/badge.svg)

[![Test it in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pysentimiento/pysentimiento/blob/master/notebooks/examples/pysentimiento_sentiment_analysis_in_spanish.ipynb)

<p>
<a href="https://console.tiyaro.ai/explore?q=pysentimiento/robertuito-&pub=pysentimiento"> <img src="https://tiyaro-public-docs.s3.us-west-2.amazonaws.com/assets/try_on_tiyaro_badge.svg"></a>
</p>




A Transformer-based library for SocialNLP tasks.

Currently supports:


| Task                                 | Languages                             |
|:---------------------                |:---------------------------------------|
| Sentiment Analysis                   | es, en, it, pt                        |
| Hate Speech Detection                | es, en, it, pt                        |
| Irony Detection                      | es, en, it, pt                        |
| Emotion Analysis                     | es, en, it                            |
| NER & POS tagging                    | es, en                                |
| Contextualized Hate Speech Detection | es                                    |
| Targeted Sentiment Analysis          | es                                    |


Just do `pip install pysentimiento` and start using it:

## Getting Started

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

emotion_analyzer = create_analyzer(task="emotion", lang="en")

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

hate_speech_analyzer.predict("Vaya guarra barata y de poca monta es XXXX!")
# returns AnalyzerOutput(output=['hateful', 'targeted', 'aggressive'], probas={hateful: 0.987, targeted: 0.978, aggressive: 0.969})
```

See [TASKS](docs/TASKS.md) for more details on the supported tasks and languages, and also for reported performance for each benchmarked model.

Also, check these notebooks with examples of how to use `pysentimiento` for each language:

- [Spanish + English](https://colab.research.google.com/github/pysentimiento/pysentimiento/blob/master/notebooks/examples/pysentimiento_sentiment_analysis_in_spanish.ipynb)
- [Italian](https://colab.research.google.com/github/pysentimiento/pysentimiento/blob/master/notebooks/examples/sentiment_analysis_in_italian.ipynb)
- [Portuguese](https://colab.research.google.com/github/pysentimiento/pysentimiento/blob/master/notebooks/examples/sentiment_analysis_in_portuguese.ipynb)

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


## Instructions for developers

0. Clone and install

```
git clone https://github.com/pysentimiento/pysentimiento
pip install poetry
poetry shell
poetry install
```

1. Run script to train models

Check [TRAIN.md](docs/TRAIN.md) for further information on how to train your models

Note: you need access to the datasets, which are not public for the time being. Send us an email to get access to them.

2. Upload models to Huggingface's Model Hub

Check ["Model sharing and upload"](https://huggingface.co/transformers/model_sharing.html) instructions in `huggingface` docs.

## License

`pysentimiento` is an open-source library. However, please be aware that models are trained with third-party datasets and are subject to their respective licenses, many of which are for non-commercial use

1. [TASS Dataset license](http://tass.sepln.org/tass_data/download.php) (License for Sentiment Analysis in Spanish, Emotion Analysis in Spanish & English)
2. [SEMEval 2017 Dataset license](https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip?file_subpath=%2F2017_English_final%2FDOWNLOAD%2FREADME.txt) (Sentiment Analysis in English)

3. [LinCE Datasets](https://ritual.uh.edu/lince/datasets) (License for NER & POS tagging)

## Suggestions and bugfixes

Please use the repository [issue tracker](https://github.com/pysentimiento/pysentimiento/issues) to point out bugs and make suggestions (new models, use another datasets, some other languages, etc)


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

Also, pleace cite related pre-trained models and datasets for the specific models you use. Check [REFERENCES](docs/REFERENCES.md) for details.

