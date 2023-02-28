# CHANGELOG

## 0.6.1rc4

- Add support for Portuguese and Italian
- Fix NER issues
- Major refactor of the codebase

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
