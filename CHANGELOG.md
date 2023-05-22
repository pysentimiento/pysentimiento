# CHANGELOG

## 0.7.1

- Allow torch >= 2.0.0

## 0.7.0

- Move torch to 2.0.0

## 0.6.7

- Add support for NER models that do not have BIO labels

## 0.6.6

- Add support for targeted sentiment analysis
- Add support for contextualized hate speech detection

## 0.6.5

- NER bugfix when sequence is too long
- NER bugfix: preprocessing was missing

## 0.6.3

- Add "USER" tag for NER and POS
- Improved NER & POS output


## 0.6.2

- Support for 3.11
- Remove unused dependencies in dev

## 0.6.1

- Add support for Portuguese and Italian
- Fix NER issues
- Major refactor of the codebase
- wandb integration and hyperparameter tuned models

## 0.5.2

- Drop unnecessary dependencies (such as sklearn)

## 0.5.1

- Add python 3.10 support (now tested on 3.8, 3.9, 3.10)
- Drop python 3.7 support

## 0.4.1

- Add NER & POS tagging in Spanish+English through the LinCE dataset.

## 0.3.2

- Fix issue with Hate Speech detection in English

## 0.3.0

- Add Hate Speech Analyzer
- Use RoBERTuito instead of BETO in Spanish models
- Refactor training scripts
- Move from Pipenv to poetry

## 0.2.5

- Fix dependency issue

## 0.2.4

- Enhance batch prediction

## 0.2.3

- Fix issue with
- Improve training scripts
- Change tokenization and special tokens

Now we use @usuario, hashtag, emoji, url

## 0.2.2

- Add surrounding spaces in emoji preprocessing

## 0.2.1

- Add missing dependencies to `setup.py`
- Add `emoji_wrapper` argument to `preprocess_tweet`

## 0.2.0

- Add Emotion Analyzer model
- Add English support
- Improve preprocessing
- Analyzers now only have a `predict` method (`predict_probas` is removed)
- Add evaluations

## 0.1.1

- Improve preprocessing
- Add tests

## 0.1.0

- Add Spanish model for Sentiment Analysis
- Add preprocessing module
