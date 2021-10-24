
## Train

```bash
python bin/train_sentiment.py "dccuchile/bert-base-spanish-wwm-cased" models/beto-sentiment-analysis/ --epochs 5 --lang es
python bin/train_sentiment.py "distilbert-base-multilingual-cased" models/distilbert-es-sentiment-analysis/ --epochs 5 --lang es
python bin/train_sentiment.py "bert-base-multilingual-cased" models/mbert-es-sentiment-analysis/ --epochs 5 --lang es

# Sentiment English

python bin/train_sentiment.py "distilbert-base-multilingual-cased" models/distilbert-en-sentiment-analysis/ --epochs 10 --lang en
python bin/train_sentiment.py "bert-base-multilingual-cased" models/mbert-en-sentiment-analysis/ --epochs 10 --lang en

python bin/train_sentiment.py "bert-base-uncased" models/bert-base-sentiment-analysis/ --epochs 10 --lang en
python bin/train_sentiment.py "roberta-base" models/roberta-base-sentiment-analysis/ --epochs 10 --lang en
python bin/train_sentiment.py "vinai/bertweet-base" models/bertweet-base-sentiment-analysis/ --epochs 10 --lang en

# Emotion Spanish

python bin/train_emotion.py "dccuchile/bert-base-spanish-wwm-cased" models/beto-emotion-analysis/ --epochs 5 --lang es
python bin/train_emotion.py "distilbert-base-multilingual-cased" models/distilbert-es-emotion-analysis/ --epochs 5 --lang es
python bin/train_emotion.py "bert-base-multilingual-cased" models/mbert-es-emotion-analysis/ --epochs 5 --lang es

# Emotion English
python bin/train_emotion.py "distilbert-base-multilingual-cased" models/distilbert-en-emotion-analysis/ --epochs 5 --lang en
python bin/train_emotion.py "bert-base-multilingual-cased" models/mbert-en-emotion-analysis/ --epochs 5 --lang en

python bin/train_emotion.py "bert-base-uncased" models/bert-base-emotion-analysis/ --epochs 5 --lang en
python bin/train_emotion.py "roberta-base" models/roberta-base-emotion-analysis/ --epochs 5 --lang en
python bin/train_emotion.py "vinai/bertweet-base" models/bertweet-base-emotion-analysis/ --epochs 5 --lang en
```

## Hate Speech

```bash
# Task A
python bin/train.py "dccuchile/bert-base-spanish-wwm-uncased" models/beto-hate-speech/ hate_speech --lang es
# Task B
# With task_b flag, you train a classifier for task B of Hate Speech using a multi-label approach
# This is more general than task A
python bin/train.py "dccuchile/bert-base-spanish-wwm-uncased" models/beto-hate-speech/ hate_speech --lang es --task_b

python bin/train.py "vinai/bertweet-base" models/bertweet-hate-speech/ hate_speech --lang en
python bin/train.py "vinai/bertweet-base" models/bertweet-hate-speech/ hate_speech --lang en --task_b
```

## Results

### Task A

```bash
output_path="evaluations/hate_speech/task_a/beto.json"
model_name="dccuchile/bert-base-spanish-wwm-uncased"
python bin/train.py --base_model $model_name\
    --lang es --task hate_speech \
    --benchmark --benchmark_output_path $output_path

output_path="evaluations/hate_speech/task_a/robertuito.json"
model_name="finiteautomata/robertuito-base-uncased"
python bin/train.py --base_model $model_name\
    --lang es --task hate_speech \
    --benchmark --benchmark_output_path $output_path
```

### English

```bash
output_path="evaluations/hate_speech/task_a/bert.json"
lang="en"
model_name="bert-base-uncased"
python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech \
    --benchmark --benchmark_output_path $output_path

output_path="evaluations/hate_speech/task_a/bertweet.json"
model_name="vinai/bertweet-base"
python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech \
    --benchmark --benchmark_output_path $output_path
```
### Task B

```bash
# Hierarchical

output_path="evaluations/hate_speech/task_b/beto.json"
python bin/train.py --base_model $model_name\
    --lang es --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr"

output_path="evaluations/hate_speech/task_b/beto-hierarchical-gamma-0.1.json"
model_name="dccuchile/bert-base-spanish-wwm-uncased"
python bin/train.py --base_model $model_name\
    --lang es --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" \
    --hierarchical --gamma 0.1


output_path="evaluations/hate_speech/task_b/robertuito-hierarchical-gamma-0.1.json"
model_name="finiteautomata/robertuito-base-uncased"
python bin/train.py --base_model $model_name\
    --lang es --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" \
    --hierarchical --gamma 0.1

output_path="evaluations/hate_speech/task_b/robertuito.json"
python bin/train.py --base_model $model_name\
    --lang es --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr"
# Combinatorial



output_path="evaluations/hate_speech/task_b/beto-combi.json"
model_name="dccuchile/bert-base-spanish-wwm-uncased"
python bin/train.py --base_model $model_name\
    --lang es --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" --combinatorial

### English


model_name="bert-base-uncased"
lang="en"
output_path="evaluations/hate_speech/task_b/bert.json"
python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr"

lang="en"
gamma=0.2
model_name="bert-base-uncased"
output_path="evaluations/hate_speech/task_b/bert-hier.json"

python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" \
    --hierarchical --gamma $gamma

output_path="evaluations/hate_speech/task_b/bert-combi.json"
python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" --combinatorial

lang="en"
model_name="vinai/bertweet-base"
output_path="evaluations/hate_speech/task_b/bertweet.json"

python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr"

model_name="vinai/bertweet-base"
output_path="evaluations/hate_speech/task_b/bertweet-hier.json"

python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" \
    --hierarchical --gamma $gamma

output_path="evaluations/hate_speech/task_b/bertweet-combi.json"
python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" --combinatorial

```

### Parameter fine-tuning

#### Spanish

```bash
for gamma in {1.0,0.9,0.8,0.7,0.5.0.4,0.3,0.2,0.1,0.05,0.01,0.005,0.00}
do
    output_path="evaluations/hate_speech/dev/emr/beto-taskb-hier-${gamma}.json"
    echo $output_path
    python bin/train.py --base_model "dccuchile/bert-base-spanish-wwm-uncased"\
        --lang es --task hate_speech --task_b \
        --benchmark --benchmark_output_path $output_path \
        --metric_for_best_model "emr" \
        --task_b --hierarchical --gamma $gamma --dev
done
```

#### English
```bash
model_name="bert-base-uncased"
for gamma in {1.0,0.9,0.8,0.7,0.5.0.4,0.3,0.2,0.1,0.05,0.01,0.005,0.00}
do
    output_path="evaluations/hate_speech/dev/emr/en/bert-taskb-hier-${gamma}.json"
    echo $output_path
    python bin/train.py --base_model $model_name \
        --lang en --task hate_speech --task_b \
        --benchmark --benchmark_output_path $output_path \
        --metric_for_best_model "emr" \
        --task_b --hierarchical --gamma $gamma --dev
done
```

```bash
model_name="vinai/bertweet-base"
for gamma in {1.0,0.9,0.8,0.7,0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.005,0.00}
do
    output_path="evaluations/hate_speech/dev/emr/en/bertweet-taskb-hier-${gamma}.json"
    echo $output_path
    python bin/train.py --base_model $model_name \
        --lang en --task hate_speech --task_b \
        --benchmark --benchmark_output_path $output_path \
        --metric_for_best_model "emr" \
        --task_b --hierarchical --gamma $gamma --dev
done
```


## Benchmarking

To run benchmarks you can use also `bin/train.py` passing the `--benchmark`

```bash
python bin/train.py --base_model "dccuchile/bert-base-spanish-wwm-uncased" --lang es --task hate_speech\
 --benchmark --task_b --benchmark_output_path evaluations/hate_speech/beto-taskb-hier.json --task_b --hierarchical
```

## Evaluation

```bash
python bin/eval_sentiment.py models/beto-sentiment-analysis/ evaluations/sentiment_beto.json --lang es
python bin/eval_sentiment.py models/distilbert-es-sentiment-analysis/ evaluations/sentiment_distilbert_es.json --lang es
python bin/eval_sentiment.py models/mbert-es-sentiment-analysis/ evaluations/sentiment_mbert_es.json --lang es


python bin/eval_sentiment.py models/mbert-en-sentiment-analysis/ evaluations/sentiment_mbert_en.json --lang en
python bin/eval_sentiment.py models/distilbert-en-sentiment-analysis/ evaluations/sentiment_distilbert_en.json --lang en
python bin/eval_sentiment.py models/bert-base-sentiment-analysis/ evaluations/sentiment_bert_base.json --lang en
python bin/eval_sentiment.py models/bertweet-base-sentiment-analysis/ evaluations/sentiment_bertweet_base.json --lang en
python bin/eval_sentiment.py models/roberta-base-sentiment-analysis/ evaluations/sentiment_roberta_base.json --lang en

# Emotion Spanish

python bin/eval_emotion.py models/beto-emotion-analysis/ evaluations/emotion_beto.json --lang es
python bin/eval_emotion.py models/distilbert-es-emotion-analysis/ evaluations/emotion_distilbert_es.json --lang es
python bin/eval_emotion.py models/mbert-es-emotion-analysis/ evaluations/emotion_mbert_es.json --lang es

# Emotion English
python bin/eval_emotion.py models/distilbert-en-emotion-analysis/ evaluations/emotion_distilbert_en.json --lang en
python bin/eval_emotion.py models/mbert-en-emotion-analysis/ evaluations/emotion_mbert_en.json --lang en

python bin/eval_emotion.py models/bert-base-emotion-analysis/ evaluations/emotion_bert_base.json --lang en
python bin/eval_emotion.py models/bertweet-base-emotion-analysis/ evaluations/emotion_bertweet_base.json --lang en
python bin/eval_emotion.py models/roberta-base-emotion-analysis/ evaluations/emotion_roberta.json --lang en
```


## Smoke test

```
python bin/train_sentiment.py "dccuchile/bert-base-spanish-wwm-cased" models/test/ --epochs 5 --limit 500 && python bin/train_sentiment.py "bert-base-uncased" models/test/ --epochs 5 --limit 500 --lang en && rm -Rf models/test/

# Emotion
python bin/train_emotion.py "dccuchile/bert-base-spanish-wwm-cased" models/test/ --lang es --epochs 3 --limit 500 && python bin/train_emotion.py "vinai/bertweet-base" models/test/ --lang en --epochs 3 --limit 500 && rm -Rf models/test/
```