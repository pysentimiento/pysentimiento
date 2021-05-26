
## Train

```bash
python bin/train_sentiment.py "dccuchile/bert-base-spanish-wwm-cased" models/beto-sentiment-analysis/ --epochs 5 --lang es


python bin/train_sentiment.py "bert-base-uncased" models/bert-base-sentiment-analysis/ --epochs 10 --lang en
python bin/train_sentiment.py "roberta-base" models/roberta-base-sentiment-analysis/ --epochs 10 --lang en

python bin/train_sentiment.py "vinai/bertweet-base" models/bertweet-base-sentiment-analysis/ --epochs 10 --lang en

# Emotion

python bin/train_emotion.py "dccuchile/bert-base-spanish-wwm-cased" models/beto-emotion-analysis/ --epochs 5 --lang es

python bin/train_emotion.py "bert-base-uncased" models/bert-base-emotion-analysis/ --epochs 5 --lang en
python bin/train_emotion.py "vinai/bertweet-base" models/bertweet-base-emotion-analysis/ --epochs 5 --lang en
```

## Evaluation

```
python bin/eval_sentiment.py models/beto-sentiment-analysis/ evaluations/sentiment_beto.json --lang es

python bin/eval_sentiment.py models/bert-base-sentiment-analysis/ evaluations/sentiment_bert_base.json --lang en
python bin/eval_sentiment.py models/bertweet-base-sentiment-analysis/ evaluations/sentiment_bertweet_base.json --lang en
python bin/eval_sentiment.py models/roberta-base-sentiment-analysis/ evaluations/sentiment_roberta_base.json --lang en

#

python bin/eval_emotion.py models/beto-emotion-analysis/ evaluations/emotion_beto.json --lang es

python bin/eval_emotion.py models/bert-base-emotion-analysis/ evaluations/emotion_bert_base.json --lang en
python bin/eval_emotion.py models/bertweet-base-emotion-analysis/ evaluations/emotion_bertweet_base.json --lang en
python bin/eval_emotion.py models/roberta-base-emotion-analysis/ evaluations/emotion_roberta.json --lang en
```

