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

#
#
# vinai/bertweet
#
#
lang="en"
model_name="vinai/bertweet-base"
output_path="evaluations/hate_speech/task_b/bertweet.json"
gamma=0.24

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
#
#
# roberta-base
#
#
lang="en"
model_name="roberta-base"
output_path="evaluations/hate_speech/task_b/roberta.json"
gamma=0.2

python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr"

output_path="evaluations/hate_speech/task_b/roberta-hier.json"

python bin/train.py --base_model $model_name\
    --lang $lang --task hate_speech --task_b \
    --benchmark --benchmark_output_path $output_path \
    --metric_for_best_model "emr" \
    --hierarchical --gamma $gamma

output_path="evaluations/hate_speech/task_b/roberta-combi.json"
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

