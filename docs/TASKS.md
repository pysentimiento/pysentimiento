# Supported tasks

`pysentimiento` supports several opinion-mining and other NLP tasks for social media,in


| Task                 | Languages                             |
|:---------------------|:--------------------------------------|
| Sentiment Analysis   | es, en, it, pt                        |
| Hate Speech Detection| es, en, it, pt                        |
| Irony Detection      | es, en, it, pt                        |
| Emotion Analysis     | es, en, it                            |
| NER & POS tagging    | es, en                                |

For each task and language, please note that we used different datasets and thus the models have some differences. Roughly, for classification tasks, `pysentimiento` have two different types of models:

- Binary or multi-class classifiers: these models return a single variable (e.g. `POS`, `NEG`, `NEU` for Sentiment Analysis in es & en, or `ironic` `not ironic` for irony detection).
- Multi-label classifiers: these models return a list of labels (e.g. `hateful`, `targeted`, `aggressive` for Hate Speech Detection in es & en; or the attacked characteristics for Hate Speech Detection), corresponding to several binary variables being predicted.

Concretely, we have the following models:

| Language      | Task                 | Output      | Classes       |
|:-----------   |:---------------------|:----------- |:--------------|
| es, en, pt    | Sentiment            | Multiclass  | POS, NEG, NEU     |
| es, en        | Emotion              | Multiclass  | anger, joy, sadness, fear, surprise, disgust, neutral|
| es, en        | Hate Speech          | Multilabel  | hateful, targeted, aggressive |
| es, en, it, pt| Irony                | Binary      | ironic, not ironic |
| it            | Sentiment            | Multilabel  | pos, neg*      |
| it            | Emotion              | Multiclass  | joy, anger, sadness, fear      |


* In Sentiment Analysis for Italian, neutral is represented by the absence of both labels. The presence of both labels is considered a mixed sentiment.


# Results
## Spanish


| Model        | Sentiment   | Emotion     | Hate Speech   | Irony       |
|:-------------|:------------|:------------|:--------------|:------------|
| BERTin       | 65.3 +- 0.5 | 50.2 +- 2.9 | 68.7 +- 1.5   | 69.3 +- 1.4 |
| BETO         | 67.2 +- 0.6 | 52.2 +- 1.4 | 73.3 +- 0.8   | 71.5 +- 0.5 |
| Electricidad | 65.3 +- 0.5 | 46.3 +- 2.3 | 71.8 +- 1.0   | 67.1 +- 2.1 |
| RoBERTa-es   | 67.3 +- 0.3 | 53.1 +- 2.2 | 73.1 +- 2.8   | 71.9 +- 0.9 |
| RoBERTuito   | 70.2 +- 0.2 | 55.3 +- 0.8 | 76.1 +- 0.5   | 74.1 +- 0.7 |


## English

| Model      | Sentiment       | Emotion        | Hate Speech       | Irony       |
|:-----------|:------------    |:------------   |:--------------    |:------------|
| BERT       | 69.6 +- 0.4     | 42.7 +- 0.6    | 56.0 +- 0.8       | 68.1 +- 2.2 |
| Electra    | 70.9 +- 0.4     | 37.2 +- 2.9    | 55.6 +- 0.6       | 71.3 +- 1.8 |
| RoBERTa    | 70.4 +- 0.3     |**45.0 +- 0.9** | 55.1 +- 0.4       | 70.4 +- 2.9 |
| RoBERTuito | 69.6 +- 0.5     | 43.0 +- 3.3    | **57.5 +- 0.2**   | 73.9 +- 1.4 |
| BERTweet   | **72.0 +- 0.4** | 43.1 +- 1.8    | **57.7 +- 0.7**   |**80.8 +- 0.7** |

## Italian

| model      | Sentiment     | Emotion       | Hate Speech       | Irony          |
|:-----------|:------------  |:------------  |:--------------    |:------------   |
| AlBERTo    | 57.8 +- 0.7   | 72.0 +- 1.3   | 88.1 +- 0.4       | 53.7 +- 0.6    |
| BERT-it    | 61.4 +- 0.9   |**73.6 +- 4.0**| **92.4 +- 0.4**   |**62.0 +- 4.4** |
| Electra-it |**62.3 +- 0.7**| 64.7 +- 7.7   | 87.8 +- 3.0       | 50.0 +- 6.5    |
| RoBERTuito | 55.2 +- 2.8   | 64.1 +- 3.0   |**92.6 +- 0.3**    | 55.6 +- 3.6    |
| UmBERTo    |**62.6 +- 1.1**| 69.7 +- 4.6   | 87.3 +- 0.4       | 60.0 +- 2.2    |

## Portuguese

| model       | Sentiment       | Hate Speech   | Irony          |
|:------------|:------------    |:--------------|----------------|
| BERT-pt     | 70.0 +- 0.3     | 64.1 +- 1.1   |   ---          |
| BERTabaporu | 73.8 +- 0.4     |**70.3 +- 3.3**|   ---          |
| BERTweet-BR | **75.3 +- 0.5** | 55.6 +- 5.5   |   ---          |
| RoBERTuito  | 71.7 +- 0.4     | 70.0 +- 2.4   |   ---          |

* Test results are not yet reported for the irony detection task
